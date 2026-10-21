# Original User Request

## 2026-10-21T10:43:28Z

Build a zero-weight, CPU/SRAM-resident 240–2,000+ FPS high-speed optical/IR video tracker, bare-metal `#![no_std]` Rust/WASM SIMD kernel, Mode B blind 240 Hz benchmark suite, and interactive WebRTC/HTML5 studio inside `c:\Users\x\Documents\antigravity\stylo` (GitHub repository: `https://github.com/sneed-and-feed/stylo`, `CC0-1.0` Public Domain) that unifies the machine-checked Lie-algebraic primitives of `formalization/OcclusionTriad.lean`, `darksnail/arc3.py`, and the 3-point discrete Morse / 1-sample `pending_impulse` causal deferral gate.

Working directory: c:\Users\x\Documents\antigravity\stylo
Integrity mode: development

## Reference Material & Existing Verification Resources

Existing codebase modules and tests in `c:\Users\x\Documents\antigravity\stylo` (seeded from `sneed-and-feed/projectdarksnail`, read-only dependencies that must remain intact with zero regressions; Python virtual environment available at `c:\Users\x\Documents\antigravity\darksnail\.venv\Scripts\python.exe`):
- `formalization/OcclusionTriad.lean` (Machine-checked Lean 4 theorems 2.1–2.7)
- `darksnail/arc3.py` (`BeltramiCallosalShell`, `RoPE2DMedoidPooler`, `TemporalClockContentSplitter`, `SkewSymmetricTriadOperator`, `SlowFastTriadCascade`, `verify_formal_theorems`)
- `darksnail/shell.py`, `darksnail/readout.py`, `darksnail/antenna.py`, `darksnail/gemma4.py`, `darksnail/hooks.py`, `darksnail/longbench.py`
- Existing 32-test suite: `tests/test_darksnail.py`, `tests/test_darksnail_arc3.py`, `tests/test_darksnail_gemma4.py`, `tests/test_hooks.py`

## Parallel Swarm Topology & Strict File Ownership (Zero Merge Conflicts)

| Agent ID | Role | Owned Files (Exclusive Write Access) | Read-Only Dependencies |
| :--- | :--- | :--- | :--- |
| **`agent-1-core`** | Core Vectorized Python Seeker & Causal Tracker | `darksnail/tracker.py` | `formalization/OcclusionTriad.lean`, `darksnail/arc3.py`, `darksnail/shell.py`, `darksnail/readout.py` |
| **`agent-2-bench`** | Mode B Blind 240 Hz Benchmark & Legacy CV Baselines | `darksnail/tracker_benchmark.py` | `darksnail/tracker.py`, `darksnail/_tracker_native.py` |
| **`agent-3-studio`** | Zero-Install WebRTC + ArrayBuffer HTML5/WebGL2 Studio | `studio/darksnail_tracker_studio.html` | `darksnail/tracker.py`, `darksnail/tracker_benchmark.py` |
| **`agent-4-rust-wasm`** | `#![no_std]` Rust SIMD (`AVX2`/`NEON`) + `.wasm` Kernel & FFI Bridge | `rust/darksnail_tracker_rs/Cargo.toml`, `rust/darksnail_tracker_rs/src/lib.rs`, `darksnail/_tracker_native.py` | `formalization/OcclusionTriad.lean`, `darksnail/tracker.py` |
| **`agent-5-referee`** | Adversarial Mode B Numerical Referee, `tracemalloc` & Parity Suite | `tests/test_darksnail_tracker.py`, `results/darksnail_tracker_audit_summary.json` | All repository files |

## Requirements

### R0. Architectural & Hardware Invariants (All Stages)
- **Zero Pretrained Neural Weights & Zero GPU Dependency:** Pure mathematical operators running inside `< 2.0 MB` active ring-buffer heap (`< 35 MB` hard `tracemalloc` ceiling) so the entire working set remains resident inside a 4 MB L2/L3 SRAM cache with zero DRAM stalls.
- **Dual Throughput Targets:**
  - **Vectorized Python Reference (`darksnail/tracker.py`):** `2.7 ms – 6.0 ms/frame` (`165 FPS – 370 FPS` single-core throughput on `128x128` streams).
  - **Compiled `#![no_std]` Rust / SIMD & WASM Kernel (`rust/darksnail_tracker_rs`):** `< 450 us/frame` (`2,000+ FPS` single-core throughput on `128x128` streams) matching Python golden fixture tensors to `<= 1.0e-7` in IEEE-754 `float32`.
- **Machine-Checked `OcclusionTriad.lean` Runtime Invariants:**
  1. `Theorem 2.1 & 2.2` (`skew_triad_swap_neg`, `occlusion_triad_breaks_symmetric_tie`): Exact skew-symmetry `S(W, i, c2, c1) = -S(W, i, c1, c2)` (`0.000e+00` residual) and deterministic tie-breaking `l_post(c1) - l_post(c2) > 0` during 2-target and multi-target crossing occlusions (`0` track-ID swaps).
  2. `Theorem 2.3 & 2.6` (`darksnail_readout_energy_conservation`, `darksnail_multistep_spherical_invariance`): Pointwise tangent orthogonality `|<v, B_W(u, v)>| <= 1.0e-8` and `0.000e+00` radial norm drift across `K in {1, 4, 8, 16}` unrolled `SlowFastTriadCascade` steps in `float32`.
  3. `Theorem 2.4 & 2.7` (`darksnail_step0_system1_homotopy`, `darksnail_multistep_step0_homotopy`): Bit-identical reduction to the unmodified baseline state trajectory (`0.000e+00` residual across all updates) when `delta_z = 0` and `InnovationSpikeGateConfig = None`.
  4. `Theorem 2.5` (`rope2d_medoid_relative_invariance` & `rope2d_arithmetic_norm_lt`): `1.0000` Tier-1 Beltrami spike recovery and exact 2D relative coordinate invariance vs. `96.94%` (`0.0306` cosine recovery) patch-averaging attenuation.

### R1. Core Vectorized Python Seeker & Causal Tracker (`darksnail/tracker.py`)
Implement `DarkSnailHighSpeedTracker` supporting both closed-loop streaming (`update(frame_t, dt)`) and batch evaluation (`track_sequence(frames, dt)`) across a 4-stage pipeline:
1. **Stage 1 — Sub-Pixel Target Locking, RoPE-2D Medoid Condensation & Whitened 3-Point Impulse Gate (`0.5 – 1.2 ms`):**
   - Condense incoming `Y8` (`uint8` / `float32`) frames into `K <= 24` spatial medoids `M(F_t) = {(x_m, y_m, s_m, tau_m)}_{m=1}^K` using `BeltramiCallosalShell` Tier-1 exterior-curl spike shielding (`1.0000` sub-pixel target preservation) and `RoPE2DMedoidPooler` Tier-2 medoids (preserving exact 2D relative displacement per Theorem 2.5).
   - **Whitened 3-Point Discrete Morse & Pullback Innovation Gate (`InnovationSpikeGateConfig`):**
     - Trigger only when the innovation squared Mahalanobis distance exceeds `chi_square_threshold(dim, nis_probability=0.999)` (e.g., `22.46` for 6-DOF, `< 0.12%` nominal false-trigger rate so nominal smoothing covariance is never degraded).
     - Pull the lookahead innovation back through the inverse transition operator `Phi^{-1}(dt)` and normalize componentwise by `sqrt(diag(R))` so position and velocity units sit on an equal dimensionless footing (`max_return_ratio = 0.35`). Raise an explicit `ValueError` on `dt < 0`, and safely defer non-finite lookahead to `"pending_impulse"` rather than failing open into `"step_maneuver"`.
   - **Dual-Mode 1-Sample Causal Deferral:** In streaming `update(frame_t, dt)`, when an unconfirmed `> 4 sigma` spike arrives, return `classification="pending_impulse"` (`update_applied=False`) and hold prior belief `(x_hat_minus, P_minus)` for 1 sample. Explicitly guard against a caller-replaced belief between steps and handle a glint-immediately-after-a-step transition. On the next sample, reject 1-frame specular glints / EW snow spikes (`classification="glint_rejected"`) or execute exact full-rank step re-initialization (`x = x_hat_minus + H^{-1} y`, `P = H^{-1} R H^{-T}`, `classification="step_maneuver"`) when a persistent high-`g` maneuver is confirmed.
2. **Stage 2 — Exogenous Strobe, HUD & Reversible Clutter Immunity (`0.4 – 0.8 ms`):**
   - Wire `TemporalClockContentSplitter` (`H_clock (+) H_nope`) using the discrete mass-continuity law `d_t rho_c + div(J_c) = sigma_c` and the **Thermodynamic Irreversibility Guard** (`reversible_mask`: `F_{t+1}(y, x) = F_{t-k}(y, x)`).
   - Quarantine 100/120 Hz PWM LED strobe flicker, stationary/scrolling HUD overlays, and reversibly swaying foliage or water ripples into `M_clock`, while isolating strictly mass-conservative translating targets (`sigma_c = 0`) in `H_nope`.
   - Support **Retroactive Track Stitching** (`_rebuild_projected_graph`) across the circular ring buffer `{F_{t-W}, ..., F_t}` whenever a periodic clutter mask updates.
3. **Stage 3 — Anti-Swap Skew-Symmetric Occlusion Crossing & Spherical Cascade (`0.8 – 1.8 ms`):**
   - In multi-target spatial collision zones (`||x_i - x_j||_2 <= r_collision`), evaluate `SkewSymmetricTriadOperator` `B_W(u_occ, v_comp) in so(C)` (Theorem 2.2) using directional approach momentum and occlusion parity to break symmetric distance/IoU ties deterministically with `0` track-ID swaps.
   - Unroll `SlowFastTriadCascade` across `K in {1, 4, 8, 16}` steps in `so(8)` coupling slow kinematic momentum (`z_H`) to fast 240 Hz frame deltas (`z_L`) via tangent rotations and exact `float32` spherical retraction (`0.000e+00` radial drift, Theorem 2.6).
4. **Stage 4 — One-Shot `SE(2)` Kinematics, `W_1^marg` Intercepts & Decoy Factorization (`1.0 – 2.2 ms`):**
   - Induce per-target `SE(2)` translation/spin operators from single frame transitions and project multi-step intercept trajectories via closed-form `O(HW)` 1D Marginal Wasserstein-1 distance (`W_1^marg`).
   - Compute the **Non-Local Causal Action Ratio** `eta_nonlocal = N_nonlocal / (N_local + N_nonlocal)` to partition **Active Causal Generators (`G_act`, `eta_nonlocal >= 0.50`)** from **Driven Passive Registers / Reflections / Decoys (`Omega_reg`)**.

### R2. Mode B Blind 240 Hz Benchmark & Legacy CV Baselines (`darksnail/tracker_benchmark.py`)
Build a reproducible, zero-leakage **Mode B Blind 240 Hz Benchmark** (`300`-trial seeded Monte Carlo sweeps across `SNR = 20 dB` down to `10 dB`, zero access to ground-truth target labels or scenario names inside the tracker) comparing:
1. `NaiveFrameDiffCentroid` (`|F_t - F_{t-1}|` + connected-component centroid + nearest-neighbor matching).
2. `PatchAvgKalmanIoU` (standard `8x8` arithmetic patch-averaged detector + constant-velocity Kalman filter + Hungarian IoU matcher + scalar NIS gate).
3. `DarkSnailHighSpeedTracker` (evaluated across Python streaming `update()`, Python batch `track_sequence()`, and the compiled Rust/SIMD native backend).

Evaluate across **5 Canonical 240 FPS Scenarios** (`128x128` and `256x256` grids, `240` frames = `1.0s` at `240 Hz`):
- **`01_subpixel_ballistic_glint`:** `1x1` and `2x2` px ballistic targets (`v = 6.5 px/frame`) under `10%` specular glare washouts and `20 dB` noise (tests Theorem 2.5 sub-pixel survival vs. `96.94%` patch-averaging dilution + Morse glint rejection).
- **`02_pwm_strobe_foliage_notch`:** Tangential target (`v = 0.5 -> 0.0 -> 1.5 px/frame`, "Doppler notch") moving through `120 Hz` PWM LED strobe flicker and reversibly swaying tree-branch clutter (`F_{t+1} = F_{t-k}`).
- **`03_symmetric_crossing_occlusion`:** 4 targets of identical intensity crossing simultaneously at the center (`90 deg` and `180 deg` head-on collisions with 5-frame full occlusion) testing Theorem 2.2 `0%` ID-swap guarantee.
- **`04_decoy_swarm_factorization`:** 2 active maneuvering command vehicles (`G_act`) surrounded by 10 slaved reflection/repeater decoys (`Omega_reg`) and a scrolling HUD countdown bar (`H_clock`).
- **`05_evasive_step_burn_maneuver`:** High-`g` instantaneous `Delta v` step maneuver interleaved with single-frame `6 sigma` optical glint spikes (testing 1-sample `pending_impulse` deferral + exact `H^{-1}` step re-initialization vs. process-noise lag, reporting post-burn RMSE at step epoch and +10 epochs alongside calibrated 6-DOF NEES).

### R3. Zero-Install WebRTC + ArrayBuffer HTML5/WebGL2 Studio (`studio/darksnail_tracker_studio.html`)
Create a standalone, zero-install, single-file dark-mode HTML5 + WebGL2/Canvas2D + WebRTC + inline WebAssembly workstation with zero external network dependencies:
- **3 Live Input Modes (100% Client-Side, Zero Server Upload):**
  1. **Built-In 240 Hz Procedural Torture Deck:** Instant 1-click switcher across Scenarios `01` through `05` with interactive sliders for SNR (`5 dB – 40 dB`), PWM Strobe Frequency (`60 / 100 / 120 Hz`), Glare Washout Rate (`0% – 20%`), and Patch-Averaging Toggle (`1x1` Native vs. `8x8` ViT Patch Blur).
  2. **Live WebRTC Camera Stream:** `navigator.mediaDevices.getUserMedia({ video: { frameRate: { ideal: 240 } } })` running the full 4-stage DarkSnail tracker live on the user's webcam feed.
  3. **Client-Side `.mp4` / `.webm` / `.y8` `ArrayBuffer` Dropzone:** Drag-and-drop local high-speed video files for frame-by-frame scrub and 240+ Hz replay.
- **Synchronized 4-Quad Tactical Viewport:**
  - **Quad A (Raw Sensor + Beltrami Shell):** Raw frame, Tier-1 Beltrami exterior-curl spikes (cyan/yellow rings), Tier-2 `RoPE2D` medoids, and rejected glare washouts (purple squares).
  - **Quad B (`H_clock (+) H_nope` Continuity Splitter):** Live heatmap of quarantined exogenous strobe/HUD/foliage clutter (`M_clock`, amber) vs. isolated mass-conservative moving targets (`H_nope`, emerald).
  - **Quad C (`so(C)` Anti-Swap Tracker vs. Legacy Kalman/IoU):** Side-by-side trajectory trails showing DarkSnail preserving exact track IDs through crossing occlusions while the `8x8` Patch+Kalman baseline drops sub-pixel targets or swaps IDs.
  - **Quad D (Causal Graph & Telemetry HUD):** Marks Active Causal Generators `G_act` (green stars) vs. Pruned Passive Decoys `Omega_reg` (red crosses), plus live readout cards for **FPS / Latency (`us/frame`)**, **Sphere Radial Drift (`0.000e+00`)**, **Active Heap (`< 2.0 MB`)**, **ID Swaps (`0`)**, and a **`verify_formal_theorems()`** badge bar (`Thm 2.1–2.7 PASS`).

### R4. `#![no_std]` Rust SIMD (`AVX2`/`NEON`) + `.wasm` Kernel & FFI Bridge (`rust/darksnail_tracker_rs/` & `darksnail/_tracker_native.py`)
Transpile the allocation-free inner loop into a zero-dependency `#![no_std]` Rust crate (`edition = "2021"`, `crate-type = ["cdylib", "rlib"]`) with a zero-copy Python `ctypes` bridge (`darksnail/_tracker_native.py`) and optional `.wasm` export:
1. **Static Ring-Buffer Arena (`< 1.5 MB` Fixed Footprint):** Pre-allocate a fixed-size `#[repr(C, align(64))]` `TrackerStateArena` struct containing the 8-frame `Y8` continuity ring buffer, `K <= 24` medoid slots, `pending_impulse` deferral slots, and `so(8)` triad cascade registers with **zero heap allocations (`no alloc`)** during `step_frame()`.
2. **SIMD Hot-Path Kernels (`aarch64` NEON + `x86_64` AVX2 + Scalar Fallback):**
   - `beltrami_exterior_curl_and_medoids_f32`: Single-pass 3x3 exterior-curl salience norm + 3-point discrete Morse washout mask + Tier-1/Tier-2 medoid extraction.
   - `temporal_continuity_clock_split_f32`: Vectorized discrete divergence `d_t rho + div(J) = sigma` and reversible-cycle mask `F_{t+1} == F_{t-k}`.
   - `skew_symmetric_triad_cascade_step_f32`: Unrolled `K in {1, 4, 8, 16}` `so(8)` tangent rotation `v_dot = B_W(u, v)` + exact `float32` unit-sphere retraction (`0.000e+00` radial drift, Theorem 2.6).
   - `innovation_spike_gate_step_f32`: `Phi^{-1}(dt)` pullback, `sqrt(diag(R))` whitening, `pending_impulse` state machine, and exact `6x6` `H^{-1}` step re-initialization.
3. **Auto-Build & Graceful Fallback (`darksnail/_tracker_native.py`):** Automatically compile the shared library via `cargo build --release` if `cargo` is present on PATH, expose zero-copy `np.ctypeslib.ndpointer` bindings to `DarkSnailHighSpeedTracker(backend="auto" | "rust" | "numpy")`, and verify `<= 1.0e-7` numerical parity against the NumPy backend.

### R5. Adversarial Mode B Numerical Audit & Parity Suite (`tests/test_darksnail_tracker.py` & `results/darksnail_tracker_audit_summary.json`)
Act as an uncompromising Mode B numerical referee:
1. **Zero Target-Leakage & Causal Purity Audit:** Inspect `darksnail/tracker.py`, `darksnail/_tracker_native.py`, and `darksnail/tracker_benchmark.py` to verify zero scenario-name branches, zero ground-truth access, and strict causal streaming in `update(frame_t, dt)` (never peeking at `frames[t+1]` before the next `update()` call).
2. **300-Trial Monte Carlo Threshold Audit:** Verify that `InnovationSpikeGateConfig` does not degrade nominal tracking RMSE when no glints or steps are present (`<= 0.12%` false trigger rate at `nis_probability = 0.999`), properly whitens mixed position/velocity noise via `sqrt(diag(R))`, and documents honest regime boundaries (including full-rank `H` requirement and `~4 sigma` threshold-hugging ambiguity rates) in `results/darksnail_tracker_audit_summary.json`.
3. **Formal Invariant, Parity & Memory Suite (`tests/test_darksnail_tracker.py`):** Test all formal theorems, Rust-to-Python parity, edge cases on the 1-sample `pending_impulse` gate, `tracemalloc` peak heap `< 2.0 MB`, and single-core throughput targets.

## Acceptance Criteria

### Formal Invariants & Numerical Parity
- [ ] `verify_formal_theorems()` passes on live tracking tensors in `tests/test_darksnail_tracker.py`: `Thm 2.1` skew-symmetry `== 0.0`, `Thm 2.2` crossing tie-break gap `> 0`, `Thm 2.3` tangent orthogonality `<= 1e-8`, `Thm 2.4 & 2.7` Step-0 / `InnovationSpikeGateConfig=None` bit-identical trajectory parity (`0.000e+00` residual across 1,000+ updates), `Thm 2.5` medoid relative invariance `<= 1e-6`, and `Thm 2.6` 16-step radial drift `== 0.000e+00`.
- [ ] `#![no_std]` Rust SIMD kernel matches the Python NumPy reference across all 4 pipeline stages within `<= 1.0e-7` in `float32`.
- [ ] `pending_impulse` 1-sample deferral gate passes all edge-case assertions: belief-replacement guard, `dt < 0` raises `ValueError`, non-finite lookahead safely defers to `"pending_impulse"`, and glint-immediately-after-step transitions are handled accurately.

### Performance, Memory & Zero-Leakage Benchmarks
- [ ] `tracemalloc` peak heap during a 1,000-frame streaming run is strictly `< 2.0 MB` (well below the `< 35 MB` repository ceiling).
- [ ] Single-core throughput on `128x128` streams exceeds `165 FPS` (`< 6.0 ms/frame`) in Python and `2,000 FPS` (`< 450 us/frame`) in the compiled Rust kernel.
- [ ] `python -m darksnail.tracker_benchmark` runs in strict Mode B (zero ground-truth or scenario-name leakage into the tracker), prints the 5-scenario comparison table (`0` crossing ID swaps on Scenario `03`), and exports `results/darksnail_tracker_audit_summary.json` including 300-trial Monte Carlo sweeps (`<= 0.12%` nominal false trigger rate at `nis_probability = 0.999`).

### Build, Regression & Studio Verification
- [ ] `cargo test --manifest-path rust/darksnail_tracker_rs/Cargo.toml --release` and `pytest tests/test_darksnail_tracker.py` pass 100% with `0` warnings.
- [ ] Existing 32-test `projectdarksnail` suite (`pytest`) passes with `0` regressions and bit-identical offline game trajectories (`ls20`, `vc33`, `ft09`).
- [ ] `studio/darksnail_tracker_studio.html` is a standalone single-file HTML5/WebGL2/WebRTC/WASM studio that opens directly via `file://` with zero external network requests and supports all 3 input modes and all 4 tactical viewport quads.
- [ ] All changes are cleanly committed and pushed to `origin main` on `https://github.com/sneed-and-feed/stylo`.
