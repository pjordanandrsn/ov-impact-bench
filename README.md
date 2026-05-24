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
- `hardware.regime` — **whether this box exercises the fix**. This is set by the
  **NEO driver version, not the GPU generation**:
  - `cl_intel_subgroup_local_block_io` **absent** (NEO 23.x+) → the fix's emulation
    path is what makes the GPU run at all. Confirmed on hardware as old as a Gen 9.5
    UHD P630 once it's on NEO 23.43 — the regime follows the driver, not the silicon.
  - **present** (older NEO) → fast path: GPU-vs-CPU delta is still real, but this box
    can't tell the bug from the fix.

## Hardware tiers

| box | runs impact A/B? | exercises the fix? |
|---|---|---|
| Any Intel GPU on **NEO 23.x+** (Arc, Battlemage, Lunar Lake — *or* a Gen 9.5 UHD P630) | ✅ | ✅ — extension dropped, emulation path exercised |
| Intel GPU on **older NEO** (extension still present) | ✅ | ❌ (fast path) — still a valid GPU-vs-CPU baseline |
| NVIDIA / Apple Silicon | ❌ (no `intel_gpu` plugin) | ❌ |

## First thing to run — anywhere

```bash
python3 ov_impact_bench.py --self-check
```

Reports detected CPU/GPU, the regime flag, and which energy backends are live on
this box. Run it on each machine to see what it can measure before trusting a
full run. The self-check is the smoke test; see **[Validated runs](#validated-runs)**
below for measured release-crash-vs-fixed-nightly results on a P630 under NEO 23.43.

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

## Population impact (honest)

The README's earlier `gen_per_day=50, devices=60k` defaults were a flat
triangulation. Walking a structured filter chain over the same evidence shifts
the affected population meaningfully and changes the framing. Every step below
is a **dial labeled as a dial** — give or take a multiplicative factor depending
on which one you push back on.

### TL;DR

- **Affected population: ~24k load-bearing midpoint; ~4–75k honest compound
  range across the dial widths.** Overwhelmingly AI-PC laptops shipped
  2024–2026 (Meteor / Lunar / Arrow Lake, Arc). The wide compound band is the
  independent-error product of the dial ranges below; some dials correlate
  (baseline-high implies LLM-share-high), which would narrow it — but that
  argument lives outside the artifact.
- **Measured floor (UHD P630):** 1.33× decode, 2.7× TTFT.
- **Physics-projected to median silicon (Iris Xe / Lunar Lake), NOT measured:**
  2–3× decode, 5–8× TTFT. The harness has run on exactly one device (the
  P630); these brackets come from a thermal-cap argument (faster GPU + power-
  capped fanless CPU widens the delta), not a measured distribution.
- **Aggregate energy delta (derived; J/token pending direct measurement on
  hwmon-capable hardware):** ~40–70 MWh/year (~$2–13k/year retail vs
  hyperscaler), using assumed J/token. The P630 validated run is energy-null
  because QTS exposes no `powercap`/hwmon.
- **Machine-time wait eliminated (derived, not bench-measured):** ~1,500–2,500
  hours/day at the population level — the measured per-device latency delta
  scaled by the filter-chain population. The bench instrument measures
  J/token and latency; this is a scaling of those, not a separate instrument.
  The *human-facing* subset (voice prompts, content generation, anything where
  a person is actually waiting at the screen) is a fraction of this total —
  probably 20–40% — because most of the silent-fallback population runs
  background / OEM-bundled workloads where nobody is blocked.
- **The dollar number is small. The UX number is the one that matters. The
  silicon-utilization number is the one Intel's platform organization would
  care about.**

### The filter chain

Starting from an OpenVINO active install base of roughly **750k** (mid of a
500k–1M central estimate; PyPI/Docker telemetry counts developers and CI but
undercounts shipped-with-the-OS — Audacity's OpenVINO music-separation plugin
since 2023, GIMP 3.0 plugins, Frigate's HA add-on, plus OEM bundles like Lenovo
Now / HP AI Companion / ASUS StoryCube riding the AI-PC wave):

| filter | rate | population |
|---|---|---|
| baseline OV deployments | — | ~750k |
| GPU plugin (vs CPU-only default) | 25% | ~190k |
| LLM workload (blended; see below) | 21.5% | ~40k |
| INT4 weight-only (dominant OV-LLM format) | 80% | ~32k |
| NEO 23.x+ (regression's driver cutoff) | 75% | **~24k** |

The **21.5%** LLM-share is a blend, not a flat global average. The population
splits roughly into a legacy ~300k bucket (Frigate / CV-heavy, ~10–15% LLM) and
an AI-PC ~450k bucket (LLM-headlined by the platform marketing, ~25–30% LLM):
`(300 × 12.5 + 450 × 27.5) / 750 ≈ 21.5%`. A homogeneous 15% global rate lands
the chain near ~17k; the AI-PC slice's 27.5% applied globally lands near ~33k.
The **24k blended figure is the load-bearing midpoint** — every downstream
number scales linearly with it.

**Honest compound width across the labeled dial ranges:**

- low corner: `500k × 0.15 × 0.15 × 0.60 × 0.60 ≈ ` **~4k**
- high corner: `1M × 0.35 × 0.28 × 0.90 × 0.85 ≈ ` **~75k**

That's roughly an order of magnitude span — not the ±25% band an earlier draft
implied. These are the bounds an independent-error model produces; they should
not be tightened without a correlation argument the artifact doesn't make.
(Some dials do correlate: baseline-high implies LLM-share-high since both
ride the AI-PC wave. Argued separately, that narrows the practical band; not
relied on here.)

### Two failure modes — only one of them gets reported

The kernel-compile failure pre-fix produced different UX depending on how
OpenVINO was invoked:

- **`device="AUTO"` / `device="HETERO:GPU,CPU"`** — the **default** in OV-GenAI,
  optimum-intel, OVMS's LLM path, and ~every OEM-bundled AI app. `cl::BuildError`
  is caught; the runtime **silently** falls back to CPU. No error message, no
  crash report, no telemetry signal — just a model that runs slower than the
  silicon it shipped on can do. Probably 70–80% of the affected ~24k.
- **`device="GPU"` explicit** — power users, benchmarks, validation runs. The
  error propagates as a hard model-load failure. Visible, easy to file. The
  smaller slice.

The pernicious case is the first one. The bug was **uninstrumented by
construction**: silent CPU fallback doesn't surface in logs or dashboards, so
the platform-level symptom ("the local AI features on this AI-PC feel
sluggish") was leaking into early reviews while the underlying cause stayed
invisible to upstream telemetry. That's why nobody had filed it — each user
just thought their model was slow.

### The validated-run number is the *floor*, not the median

The 2.7× TTFT / 1.33× decode in the Validated Runs table came from a UHD P630
(Gen 9.5, 24 EUs) — the *weakest* silicon in the affected population.
Population-weighted, the median sits at Iris Xe / Meteor / Lunar Lake, where
the GPU side scaled while the CPU AVX path is power/thermal-capped in fanless
laptops. On Lunar Lake (~67 GPU TOPS available to OV vs a power-capped CPU),
the population-weighted TTFT recovery is **physics-projected** at 5–8× and
decode at 2–3×.

**Caveat the projection deserves:** the harness has run on exactly one device.
The "floor, not the median" framing rests on a thermal-cap physics argument
(faster GPU + power-capped fanless CPU widens the delta as you move up the
silicon ladder), not on a measured distribution across hardware. Converting
the assertion into a demonstration is a single second-hardware run away —
a used X1 Carbon Gen 11 (Iris Xe), a Lunar Lake AI-PC laptop, or a borrowed
Arc desktop would do it. That belongs on the punch list alongside the
wall-plug energy run on hwmon-capable hardware.

### Dial anchors

- **750k baseline.** PyPI install counts (hundreds of thousands/week, CI-heavy),
  Docker pulls (millions, CI-heavy), Intel marketing on AI-PC volume (~100M
  units by end-2025, aspirational), Audacity (tens of millions of installs, the
  largest single consumer-scale OV consumer), OEM bundle activation rates (low
  single-digit %). Range 500k–1M.
- **25% GPU plugin.** OV defaults to CPU; GPU concentrates in iGPU laptops, Arc
  desktops, the AI-PC wave. Range 15–35%.
- **21.5% LLM (blended).** Frigate/CV dominates legacy OV; LLM is the AI-PC
  headliner. Range 15–28% global.
- **80% INT4.** HF OpenVINO hub's LLM artifacts lean heavily INT4 weight-only.
  Range 60–90%.
- **75% NEO 23.x+.** Drivers lag on Ubuntu LTS / Debian stable; faster on AI-PC
  laptops where the silicon's new. Range 60–85%.

### What this section does *not* claim

The arc above scales bench measurements (J/token, TTFT, throughput) to a
population using explicit dials. It does not denominate the value in
engineering hours, knowledge-worker wages, or "careers' worth" of human
waiting — those are separate Fermi exercises that an earlier draft of this
section attempted, and they don't follow from the instrument. The bench
measures machine latency and per-token energy on a device. Translating
machine latency into human productive output requires assumptions about
attention/interruption conversion that live entirely outside the bench, and
multiplying fragmented inference waits by a knowledge-worker wage rate
overstates by mixing units. The defensible artifact-level claim is the
throughput + energy scaling already in the TL;DR: ~1.5–2.5k hours/day of
machine-time wait eliminated at the population level, derived from the
measured TTFT delta; ~40–70 MWh/year aggregate energy delta pending direct
J/token measurement on hwmon-capable hardware. Anything beyond that — user
productivity gains, platform-reputation effects, fleet-engagement second-order
impacts — is real, but it lives outside the instrument and should be argued
separately, not bolted onto the bench's output.

### The old `devices=60k` default was right for the wrong reasons

The pre-revision flat extrapolation landed about 2–4× the careful estimate. Two
errors partially canceled: the baseline was inflated by PyPI/Docker noise, and
the AI-PC laptop population was undercounted because PyPI can't see
shipped-with-the-OS apps. Once the chain is done honestly the truth lands in
the same neighborhood, but you can defend each step. The 60k will likely be
closer to right than wrong by the time the 2024–2026 AI-PC silicon base is
fully baked in.

## Honesty notes

- The projection's `j_per_token_saved` is measured; `devices_assumed` and
  `gen_per_day` are **your dials** — see [Population impact (honest)](#population-impact-honest)
  above for a structured filter-chain estimate (~24k load-bearing, range
  17–30k). The aggregate $/MWh inherits all the volume uncertainty; only the
  per-token physics is observed.
- A single box measures one device pairing. Trends across hardware/driver/model
  are what make the dataset valuable — hence the append-only log + the timer.
