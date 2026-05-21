# ov-impact-bench

Measures the **real** GPU-vs-CPU-fallback impact of OpenVINO LLM inference on
Intel hardware — the per-inference latency / energy / throughput delta that the
`openvinotoolkit/openvino` `__local` block-read fix (PRs #35661, #35712) unlocks.

It exists to replace one estimate with one measurement. Every "$/Watts saved"
projection bottoms out on a single unknown — **Joules per token saved by staying
on the GPU instead of falling back to CPU**. This harness measures that number
directly; only the *volume* dial (devices × inferences/day) stays a guess.

## What it records, per run

- `tok_s_median`, `ttft_ms_median`, `tpot_ms_median` — from OpenVINO GenAI's
  built-in `PerfMetrics` (not hand-timed).
- `j_per_token_mean`, `active_power_w_mean` — from the best available energy backend.
- `cpu_busy_pct_during_run` — the "does it make the machine unusable" axis.
- `delta_gpu_vs_cpu` — speedup ×, energy-ratio ×, **J/token saved**, and a
  projection that plugs the measured J/token into your volume dials.
- `hardware.regime` — **whether this box exercises the fix**:
  - `cl_intel_subgroup_local_block_io` **absent** → modern NEO (Arc / Battlemage /
    Lunar Lake): the fix's emulation path is what makes the GPU run at all.
  - **present** → fast path (e.g. UHD P630): GPU-vs-CPU delta is still real, but
    this box can't tell the bug from the fix.

## Hardware tiers

| box | runs impact A/B? | exercises the fix? |
|---|---|---|
| Arc / Battlemage / Lunar Lake (NEO 23.x+) | ✅ | ✅ — the real target |
| Older Intel iGPU (UHD P630, etc.) | ✅ | ❌ (fast path) — still a valid GPU-vs-CPU baseline |
| NVIDIA / Apple Silicon | ❌ (no `intel_gpu` plugin) | ❌ |

## First thing to run — anywhere

```bash
python3 ov_impact_bench.py --self-check
```

Reports detected CPU/GPU, the regime flag, and which energy backends are live on
this box. Run it on each machine to see what it can measure before trusting a
full run. **This script has not yet been validated on affected hardware** — the
self-check is the smoke test.

## Energy backends (auto-detected, best → convenient)

1. **HA wall plug** — true wall draw incl. discrete GPU + PSU losses;
   topology-independent and the most honest number. Set:
   ```bash
   export OVB_HA_URL=https://homeassistant.local:8123
   export OVB_HA_TOKEN=$(cat ~/.config/ovb/ha-token)   # never paste tokens inline
   python3 ov_impact_bench.py --ha-entity sensor.ai_pc_plug_power
   ```
2. **Intel RAPL** — CPU package/psys energy. Usually root-only (`energy_uj` is
   `0400 root` on modern kernels) → run with `sudo` or use the HA backend.
   Captures iGPU (inside the package) but **not** a discrete Arc.
3. **GPU hwmon** — discrete-GPU energy/power via `/sys/class/drm/card*/.../hwmon`.

For a discrete Arc the cleanest total is **HA wall plug** (RAPL misses the card);
for an iGPU, RAPL-package already includes it.

## Full run

```bash
python3 ov_impact_bench.py \
  --model OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov \
  --devices GPU,CPU --iters 5 --max-new-tokens 256 \
  --ha-entity sensor.ai_pc_plug_power
```

Appends one JSON record per run to `~/ov-impact-bench/results.jsonl`, so runs
accumulate into a trend you can chart over driver/model/hardware changes.

If the GPU device **fails to compile** (the pre-fix failure mode), that device's
entry is `{"error": ...}` — which is itself the result: it's the bug, captured.

## Run from now on

A weekly systemd-user timer is in `systemd/`. Install on the box with the GPU:

```bash
cp systemd/ov-impact-bench.* ~/.config/systemd/user/
# edit the .service ExecStart paths + OVB_HA_* if using the wall plug
systemctl --user daemon-reload
systemctl --user enable --now ov-impact-bench.timer
systemctl --user list-timers ov-impact-bench.timer
```

## Validated runs

First end-to-end validation, 2026-05-21, on the home-lab QNAP (Xeon W-1250 +
UHD P630 iGPU passed into a Container Station container). The P630 under NEO
23.43 advertises **no** `cl_intel_subgroup_local_block_io` — i.e. it sits in the
exact modern-NEO regime that PR
[openvinotoolkit/openvino#35712](https://github.com/openvinotoolkit/openvino/pull/35712)
fixes. Same hardware, same model (`TinyLlama-1.1B-Chat-int4-ov`), two wheels:

| | release `2026.1.0` (pre-fix) | nightly `2026.3.0.dev20260520` (with #35712) |
|---|---|---|
| GPU kernel compile | ❌ `cl::BuildError: clBuildProgram` | ✅ compiles |
| GPU decode | — (falls back to CPU) | 18.4 tok/s |
| CPU decode | 13.9 tok/s | 13.9 tok/s |
| **GPU speedup** | — | **1.33×** |
| **TTFT** | CPU 355 ms | **GPU 133 ms (2.7× faster)** |

On the *weakest* affected GPU (24-EU Gen 9.5), the fix recovers ~1.33× decode and
~2.7× TTFT versus the CPU fallback it prevents; Arc/Lunar Lake would widen that.
`j_per_token` is null here — QTS exposes no powercap/hwmon, so the energy axis
needs the HA wall-plug backend on this box. The release-wheel crash is captured
as a structured `{error, diagnosis}` record (subprocess isolation), so the
bug-vs-fix delta is reproducible from `results.jsonl`.

## Honesty notes

- The projection's `j_per_token_saved` is measured; `devices_assumed` and
  `gen_per_day` are **your dials** — defaults (~60k devices, 50 gen/day) are the
  triangulated central estimate, not data. The aggregate $/MWh inherits all the
  volume uncertainty; only the per-token physics is observed.
- A single box measures one device pairing. Trends across hardware/driver/model
  are what make the dataset valuable — hence the append-only log + the timer.
