#!/usr/bin/env python3
"""
ov-impact-bench — measure the real GPU-vs-CPU-fallback impact of OpenVINO LLM
inference on Intel hardware.

Background
----------
OpenVINO GPU kernels (LoRA / MoE / fully_connected GEMV) failed to compile on
Intel Compute Runtime (NEO) 23.x+ because the __local-pointer overloads of
intel_sub_group_block_read* moved to the cl_intel_subgroup_local_block_io
extension. Under AUTO/HETERO that forced whole-model fallback to CPU. PRs
openvinotoolkit/openvino#35661 and #35712 fixed it (the _sub_group_block_read_slm*
family in sub_group_block_read.cl).

This harness measures what that fix is worth, per inference, on real silicon:
the same OpenVINO model is run once on GPU and once forced to CPU (the fallback
counterfactual), and we record latency, throughput, energy/token, power, and
CPU utilisation for each. The only thing the whole "$/Watts" estimate could not
pin was the per-token delta — this measures it. Volume (devices x inferences)
remains the user's dial; everything else here is observed.

It also reports WHICH regime the GPU is in:
  * cl_intel_subgroup_local_block_io ABSENT  -> modern NEO, the fix's emulation
    path is what makes the GPU run at all (Arc / Battlemage / Lunar Lake).
  * cl_intel_subgroup_local_block_io PRESENT -> fast path; the GPU-vs-CPU delta
    is still real, but this box can't distinguish the bug from the fix
    (e.g. UHD P630 / older iGPUs).

Energy backends (best -> convenient), auto-detected, recorded side by side:
  1. HA wall plug   — true wall draw incl. dGPU + PSU losses. Most robust,
                      topology-independent. Needs OVB_HA_URL + OVB_HA_TOKEN env
                      and --ha-entity. (You already run Home Assistant.)
  2. Intel RAPL     — CPU package/psys energy via /sys/class/powercap. Usually
                      root-only on modern kernels. Captures iGPU (in package)
                      but NOT a discrete Arc.
  3. GPU hwmon      — discrete-GPU energy/power via /sys/class/drm/card*/.../hwmon.

NOTE: this script has not been run on affected hardware yet. Run `--self-check`
first on any box to see what it can actually measure there before trusting a run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

DEFAULT_MODEL = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"  # #2 on the HF OV-org list; small + fast
DEFAULT_PROMPTS = [
    "Explain how a CPU pipeline hazard is resolved, in three sentences.",
    "Write a haiku about shared local memory on a GPU.",
    "Summarise the tradeoffs between INT4 and FP16 weight quantization.",
    "What is a subgroup block read and why does address space matter?",
]


# --------------------------------------------------------------------------- #
# Hardware / regime detection
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stdout
    except Exception:
        return ""


def detect_hardware() -> dict:
    hw: dict = {
        "host": platform.node(),
        "kernel": platform.release(),
        "cpu": "unknown",
        "intel_gpu": None,
        "neo_driver": None,
        "local_block_io": None,  # True/False/None(unknown)
        "regime": "unknown",
    }
    # CPU model
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    hw["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        hw["cpu"] = platform.processor() or "unknown"

    # GPU + extensions via clinfo (best source for the extension flag)
    clinfo = _run(["clinfo"]) if shutil.which("clinfo") else ""
    if clinfo:
        m = re.search(r"Device Name\s+(Intel.*)", clinfo)
        if m:
            hw["intel_gpu"] = m.group(1).strip()
        m = re.search(r"Driver Version\s+([\d.]+)", clinfo)
        if m:
            hw["neo_driver"] = m.group(1).strip()
        hw["local_block_io"] = "cl_intel_subgroup_local_block_io" in clinfo

    # Fallback: ask OpenVINO for the GPU name even if clinfo is absent
    if hw["intel_gpu"] is None:
        try:
            import openvino as ov  # noqa: PLC0415

            core = ov.Core()
            if "GPU" in core.available_devices:
                hw["intel_gpu"] = core.get_property("GPU", "FULL_DEVICE_NAME")
        except Exception:
            pass

    if hw["local_block_io"] is True:
        hw["regime"] = "fast-path (extension present; cannot distinguish bug vs fix)"
    elif hw["local_block_io"] is False:
        hw["regime"] = "modern-NEO emulation (THIS is where the fix is load-bearing)"
    return hw


# --------------------------------------------------------------------------- #
# Energy backends — each returns Joules consumed across a sampled window.
# --------------------------------------------------------------------------- #
class EnergySampler:
    """Base: subclasses implement start()/stop()->joules and .available()."""

    name = "none"

    def available(self) -> bool:
        return False

    def start(self) -> None:  # pragma: no cover - interface
        ...

    def stop(self) -> float:  # pragma: no cover - interface
        return float("nan")


class RaplSampler(EnergySampler):
    """CPU package/psys energy via Intel RAPL. Often root-only."""

    name = "rapl"

    def __init__(self) -> None:
        self.domains = {}  # path -> (name, max_uj)
        for energy_path in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"):
            base = os.path.dirname(energy_path)
            try:
                with open(os.path.join(base, "name")) as fh:
                    dname = fh.read().strip()
                with open(os.path.join(base, "max_energy_range_uj")) as fh:
                    dmax = int(fh.read().strip())
                # probe readability now so available() is honest
                with open(energy_path) as fh:
                    fh.read()
                self.domains[energy_path] = (dname, dmax)
            except OSError:
                continue
        self._start = {}

    def available(self) -> bool:
        return bool(self.domains)

    def _read(self) -> dict:
        out = {}
        for path in self.domains:
            try:
                with open(path) as fh:
                    out[path] = int(fh.read().strip())
            except OSError:
                out[path] = None
        return out

    def start(self) -> None:
        self._start = self._read()

    def stop(self) -> float:
        end = self._read()
        # Prefer a 'psys' domain (whole-SoC) if present; else sum 'package-*'.
        psys = [p for p, (n, _) in self.domains.items() if "psys" in n.lower()]
        pkgs = [p for p, (n, _) in self.domains.items() if "package" in n.lower()]
        chosen = psys or pkgs or list(self.domains)
        total_uj = 0
        for path in chosen:
            s, e = self._start.get(path), end.get(path)
            if s is None or e is None:
                continue
            _, dmax = self.domains[path]
            delta = e - s
            if delta < 0:  # counter wraparound
                delta += dmax
            total_uj += delta
        return total_uj / 1e6  # uJ -> J


class GpuHwmonSampler(EnergySampler):
    """Discrete-GPU energy via drm hwmon (energy1_input uJ, or power1_average uW)."""

    name = "gpu_hwmon"

    def __init__(self) -> None:
        self.energy_file = None
        self.power_file = None
        for card in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*"):
            e = os.path.join(card, "energy1_input")
            p = os.path.join(card, "power1_average")
            if os.path.exists(e):
                self.energy_file = e
                break
            if os.path.exists(p) and self.power_file is None:
                self.power_file = p
        self._e0 = None
        self._t0 = None
        self._psamples: list[float] = []
        self._stop = threading.Event()
        self._thr = None

    def available(self) -> bool:
        return bool(self.energy_file or self.power_file)

    def _poll_power(self) -> None:
        while not self._stop.wait(0.1):
            try:
                with open(self.power_file) as fh:
                    self._psamples.append(int(fh.read().strip()) / 1e6)  # uW -> W
            except OSError:
                pass

    def start(self) -> None:
        self._t0 = time.monotonic()
        if self.energy_file:
            with open(self.energy_file) as fh:
                self._e0 = int(fh.read().strip())
        elif self.power_file:
            self._psamples = []
            self._stop.clear()
            self._thr = threading.Thread(target=self._poll_power, daemon=True)
            self._thr.start()

    def stop(self) -> float:
        dt = time.monotonic() - self._t0
        if self.energy_file:
            with open(self.energy_file) as fh:
                return (int(fh.read().strip()) - self._e0) / 1e6  # uJ -> J
        if self.power_file and self._thr:
            self._stop.set()
            self._thr.join(timeout=1)
            if self._psamples:
                return statistics.mean(self._psamples) * dt  # avg W * s = J
        return float("nan")


class HAWallSampler(EnergySampler):
    """True wall draw via a Home Assistant power sensor (Watts). Topology-proof."""

    name = "ha_wall"

    def __init__(self, entity: str | None) -> None:
        self.entity = entity
        self.url = os.environ.get("OVB_HA_URL", "").rstrip("/")
        self.token = os.environ.get("OVB_HA_TOKEN", "")
        self._samples: list[float] = []
        self._t0 = None
        self._stop = threading.Event()
        self._thr = None

    def available(self) -> bool:
        return bool(self.entity and self.url and self.token)

    def _read_w(self) -> float | None:
        req = urllib.request.Request(
            f"{self.url}/api/states/{self.entity}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return float(json.load(r)["state"])
        except Exception:
            return None

    def _poll(self) -> None:
        while not self._stop.wait(0.5):
            w = self._read_w()
            if w is not None:
                self._samples.append(w)

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._samples = []
        self._stop.clear()
        self._thr = threading.Thread(target=self._poll, daemon=True)
        self._thr.start()

    def stop(self) -> float:
        dt = time.monotonic() - self._t0
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)
        if self._samples:
            return statistics.mean(self._samples) * dt
        return float("nan")


def build_samplers(ha_entity: str | None) -> list[EnergySampler]:
    cands = [HAWallSampler(ha_entity), RaplSampler(), GpuHwmonSampler()]
    return [s for s in cands if s.available()]


# --------------------------------------------------------------------------- #
# CPU utilisation (lightweight /proc/stat delta over the run)
# --------------------------------------------------------------------------- #
def _cpu_busy_fraction(prev: list[int] | None):
    try:
        with open("/proc/stat") as fh:
            parts = [int(x) for x in fh.readline().split()[1:]]
    except OSError:
        return None, None
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
    total = sum(parts)
    if prev is None:
        return None, (idle, total)
    pidle, ptotal = prev
    dt = total - ptotal
    if dt <= 0:
        return None, (idle, total)
    return 100.0 * (1 - (idle - pidle) / dt), (idle, total)


# --------------------------------------------------------------------------- #
# Model resolution — accept a local OV-IR dir OR an HF repo id (auto-download).
# --------------------------------------------------------------------------- #
def resolve_model(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    print(f"[ov-impact-bench] downloading {model} (first run only) ...",
          file=sys.stderr)
    return snapshot_download(model)


# --------------------------------------------------------------------------- #
# Benchmark one device
# --------------------------------------------------------------------------- #
def bench_device(model: str, device: str, prompts: list[str], iters: int,
                 max_new: int, ha_entity: str | None) -> dict:
    import openvino_genai as ov_genai  # noqa: PLC0415

    pipe = ov_genai.LLMPipeline(model, device)
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new
    cfg.apply_chat_template = False
    # Force exactly max_new decode tokens so TPOT/throughput are well-defined
    # (otherwise an early EOS yields <=1 token and OpenVINO returns -1 sentinels).
    cfg.ignore_eos = True

    # Warmup (model load + first-run JIT) excluded from measurement.
    # List input -> DecodedResults (carries .perf_metrics); a bare str input
    # returns a plain str with no metrics.
    pipe.generate([prompts[0]], cfg)

    ttfts, tpots, thrus, jpts, powers = [], [], [], [], []
    _, cpu_prev = _cpu_busy_fraction(None)
    samplers = build_samplers(ha_entity)

    wall0 = time.monotonic()
    for _ in range(iters):
        for prompt in prompts:
            for s in samplers:
                s.start()
            t0 = time.monotonic()
            res = pipe.generate([prompt], cfg)
            dt = time.monotonic() - t0
            energies = {s.name: s.stop() for s in samplers}

            pm = res.perf_metrics
            ntok = pm.get_num_generated_tokens()
            ttfts.append(pm.get_ttft().mean)        # ms
            tpots.append(pm.get_tpot().mean)        # ms/token
            thrus.append(pm.get_throughput().mean)  # tok/s
            # J/token from the best available backend (samplers ordered best-first)
            if samplers and ntok:
                j = next((energies[s.name] for s in samplers
                          if energies[s.name] == energies[s.name]), float("nan"))
                if j == j:  # not NaN
                    jpts.append(j / ntok)
                    powers.append(j / dt)  # avg W during this generation
    wall = time.monotonic() - wall0
    busy, _ = _cpu_busy_fraction(cpu_prev)

    def med(xs):
        return round(statistics.median(xs), 4) if xs else None

    return {
        "device": device,
        "energy_backend": samplers[0].name if samplers else None,
        "tok_s_median": med(thrus),
        "ttft_ms_median": med(ttfts),
        "tpot_ms_median": med(tpots),
        "j_per_token_mean": round(statistics.mean(jpts), 4) if jpts else None,
        "active_power_w_mean": round(statistics.mean(powers), 2) if powers else None,
        "cpu_busy_pct_during_run": round(busy, 1) if busy is not None else None,
        "wall_s": round(wall, 2),
        "samples": len(thrus),
    }


# --------------------------------------------------------------------------- #
# Projection — anchor the $/Watts estimate on the ONE measured number.
# --------------------------------------------------------------------------- #
def project(j_per_token_saved: float, gen_per_day: int, tokens_per_gen: int,
            devices: int, price_kwh: float) -> dict:
    wh_per_gen = j_per_token_saved * tokens_per_gen / 3600.0
    per_device_kwh_yr = wh_per_gen * gen_per_day * 365 / 1000.0
    agg_kwh_yr = per_device_kwh_yr * devices
    return {
        "_note": "J/token_saved is MEASURED; gen_per_day/devices are YOUR dials (unmeasured).",
        "j_per_token_saved": round(j_per_token_saved, 4),
        "wh_per_generation": round(wh_per_gen, 4),
        "per_device_kwh_yr": round(per_device_kwh_yr, 2),
        "devices_assumed": devices,
        "aggregate_mwh_yr": round(agg_kwh_yr / 1000.0, 1),
        "aggregate_usd_yr": round(agg_kwh_yr * price_kwh, 0),
        "aggregate_kw_continuous": round(agg_kwh_yr / 8760.0, 2),
    }


# --------------------------------------------------------------------------- #
# Device isolation — a native clBuildProgram failure (the pre-fix bug) raises a
# C++ cl::BuildError -> std::terminate -> SIGABRT, which Python try/except cannot
# catch. So each device is benchmarked in a child process; a crashed child is
# recorded as a structured result instead of aborting the whole run.
# --------------------------------------------------------------------------- #
def load_prompts(prompts_file: str | None) -> list[str]:
    if prompts_file:
        with open(prompts_file) as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    return DEFAULT_PROMPTS


def run_device_isolated(model_path: str, dev: str, args) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker-device", dev,
           "--model", model_path, "--iters", str(args.iters),
           "--max-new-tokens", str(args.max_new_tokens)]
    if args.ha_entity:
        cmd += ["--ha-entity", args.ha_entity]
    if args.prompts_file:
        cmd += ["--prompts-file", args.prompts_file]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"device": dev, "error": "timeout (>1800s)"}
    for line in reversed(p.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    rc = p.returncode
    return {
        "device": dev,
        "error": f"worker exited rc={rc}"
                 + (f" (killed by signal {-rc})" if rc < 0 else ""),
        "stderr_tail": p.stderr.strip().splitlines()[-4:],
        "diagnosis": ("native clBuildProgram / std::terminate — consistent with the "
                      "pre-fix __local block-read kernel compile failure on modern NEO "
                      "(this IS the bug #35712 fixes)"),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--devices", default="GPU,CPU",
                    help="comma list; A/B is first-vs-last")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--ha-entity", default=os.environ.get("OVB_HA_ENTITY"),
                    help="HA power sensor entity_id for wall-draw truth")
    ap.add_argument("--out", default=os.path.expanduser("~/ov-impact-bench/results.jsonl"))
    ap.add_argument("--prompts-file", help="newline-delimited prompts; else built-in")
    # projection dials (defaults are the triangulated central estimates from analysis)
    ap.add_argument("--gen-per-day", type=int, default=50)
    ap.add_argument("--devices-assumed", type=int, default=60000)
    ap.add_argument("--price-kwh", type=float, default=0.16)
    ap.add_argument("--self-check", action="store_true",
                    help="report detected hardware + energy backends, then exit")
    ap.add_argument("--worker-device", help=argparse.SUPPRESS)  # internal: 1-device child
    args = ap.parse_args()

    # Worker mode: benchmark exactly one device in this isolated process.
    if args.worker_device:
        try:
            res = bench_device(resolve_model(args.model), args.worker_device,
                               load_prompts(args.prompts_file), args.iters,
                               args.max_new_tokens, args.ha_entity)
        except Exception as exc:
            res = {"device": args.worker_device, "error": repr(exc)}
        print(json.dumps(res))
        return 0

    hw = detect_hardware()

    if args.self_check:
        samplers = build_samplers(args.ha_entity)
        print(json.dumps({
            "hardware": hw,
            "energy_backends_available": [s.name for s in samplers] or ["NONE"],
            "rapl_note": "if RAPL missing but expected, rerun with sudo (energy_uj is root-only)",
            "ready": bool(samplers) and hw["intel_gpu"] is not None,
        }, indent=2))
        return 0

    try:
        model_path = resolve_model(args.model)
    except Exception as exc:
        print(f"[ov-impact-bench] model resolve failed: {exc!r}", file=sys.stderr)
        return 1

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    runs = {}
    for dev in devices:
        print(f"[ov-impact-bench] benchmarking {dev} (isolated worker) ...", file=sys.stderr)
        runs[dev] = run_device_isolated(model_path, dev, args)

    # A/B delta: first device (GPU) vs last (CPU fallback)
    gpu, cpu = runs.get(devices[0], {}), runs.get(devices[-1], {})
    delta = {}
    if gpu.get("tok_s_median") and cpu.get("tok_s_median"):
        delta["speedup_x"] = round(gpu["tok_s_median"] / cpu["tok_s_median"], 2)
    if gpu.get("j_per_token_mean") and cpu.get("j_per_token_mean"):
        saved = cpu["j_per_token_mean"] - gpu["j_per_token_mean"]
        delta["energy_ratio_x"] = round(cpu["j_per_token_mean"] / gpu["j_per_token_mean"], 2)
        delta["j_per_token_saved"] = round(saved, 4)
        delta["projection"] = project(saved, args.gen_per_day, args.max_new_tokens,
                                       args.devices_assumed, args.price_kwh)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "hardware": hw,
        "runs": runs,
        "delta_gpu_vs_cpu": delta,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(record) + "\n")

    print(json.dumps(record, indent=2))
    print(f"\n[ov-impact-bench] appended to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
