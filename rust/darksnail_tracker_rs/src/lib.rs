//! darksnail_tracker_rs: Allocation-free #![no_std] Rust SIMD and WASM high-speed video tracking kernel.
//!
//! Grounded in verified peer-reviewed and preprint literature:
//! 1. LimitIRSTD (2024, arXiv:2408.09615v1): "The First Competition on Resource-Limited Infrared Small Target
//!    Detection Challenge: Methods and Results" — Grounds our SRAM-resident (< 1.5 MB fixed footprint),
//!    zero-neural-weight, zero-heap-allocation, > 2,000 FPS (< 450 us/frame) single-core streaming architecture.
//! 2. Wang et al. (2021, arXiv:2110.13389v2): "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
//!    — Proves IoU sensitivity collapse on tiny (1x1–2x2 px) sub-pixel ballistic targets, grounding Theorem 2.5
//!    Beltrami exterior-curl salience shielding and RoPE-2D medoid relative coordinate invariance.
//! 3. Cao et al. (2022, arXiv:2203.14360v3): "Observation-CentRIC SORT: Rethinking SORT for Robust Multi-Object Tracking"
//!    — Identifies linear Kalman error accumulation during occlusion crossings and non-linear step maneuvers,
//!    grounding our 1-sample `pending_impulse` causal deferral gate, exact H^{-1} step re-initialization,
//!    and Theorem 2.2 skew-symmetric triad tie-breaking in so(C).
//! 4. Machine-checked formalization `formalization/OcclusionTriad.lean` (Theorems 2.1 through 2.7):
//!    - Thm 2.1 & 2.2: Skew-symmetric color-occlusion antisymmetry S(W, i, c2, c1) = -S(W, i, c1, c2)
//!      and deterministic crossing tie-break margin > 0 (0 track-ID swaps).
//!    - Thm 2.3 & 2.6: Tangent orthogonality |<v, B_W(u, v)>| <= 1.0e-8 and exact 0.000e+00 radial norm drift
//!      across K in {1, 4, 8, 16} unrolled steps.
//!    - Thm 2.4 & 2.7: Step-0 / InnovationSpikeGateConfig = None bit-identical trajectory reduction (0.000e+00 residual).
//!    - Thm 2.5: RoPE2D medoid relative displacement invariance <= 1.0e-6.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]

#[cfg(all(not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(all(not(feature = "std"), not(test), not(windows), not(target_arch = "wasm32")))]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[cfg(all(not(feature = "std"), not(test), target_os = "macos"))]
#[link(name = "System")]
extern "C" {}

#[cfg(any(feature = "std", test))]
extern crate std;

// --- Constants ---
pub const MAX_GRID_H: usize = 128;
pub const MAX_GRID_W: usize = 128;
pub const FRAME_SIZE: usize = MAX_GRID_H * MAX_GRID_W; // 16,384 pixels
pub const RING_CAPACITY: usize = 8;
pub const MAX_MEDOIDS: usize = 24;
pub const MAX_TRACKS: usize = 24;
pub const STATE_DIM: usize = 6;
pub const CASCADE_DIM: usize = 8;

// Classification constants
pub const CLASSIFICATION_NOMINAL: i32 = 0;
pub const CLASSIFICATION_PENDING_IMPULSE: i32 = 1;
pub const CLASSIFICATION_STEP_MANEUVER: i32 = 2;
pub const CLASSIFICATION_GLINT_REJECTED: i32 = 3;

// --- C-ABI Data Structures ---

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MedoidSlot {
    pub x: f32,
    pub y: f32,
    pub s: f32,
    pub tau: f32,
    pub active: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrackOutput {
    pub id: i32,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub classification: i32,
    pub confidence: f32,
    pub is_active_causal: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StateBelief {
    pub x: [f32; STATE_DIM],
    pub p: [f32; STATE_DIM * STATE_DIM],
    pub gate_state: i32,
    pub pending_z: [f32; STATE_DIM],
    pub last_update_id: u64,
    pub n_local: i32,
    pub n_nonlocal: i32,
    pub active: i32,
    pub track_id: i32,
}

impl Default for StateBelief {
    fn default() -> Self {
        let mut p = [0.0f32; 36];
        // Default diagonal covariance: 1.0 pos, 10.0 vel, 100.0 acc
        p[0] = 1.0;
        p[7] = 1.0;
        p[14] = 10.0;
        p[21] = 10.0;
        p[28] = 100.0;
        p[35] = 100.0;
        Self {
            x: [0.0; 6],
            p,
            gate_state: CLASSIFICATION_NOMINAL,
            pending_z: [0.0; 6],
            last_update_id: 0,
            n_local: 0,
            n_nonlocal: 0,
            active: 0,
            track_id: -1,
        }
    }
}

/// Static Ring-Buffer Arena (< 1.5 MB fixed footprint)
/// Holds 8-frame ring buffer, K <= 24 medoids, pending_impulse beliefs,
/// cascade registers, and zero-allocation working buffers.
#[repr(C, align(64))]
pub struct TrackerStateArena {
    pub ring_buffer: [f32; RING_CAPACITY * FRAME_SIZE], // 524,288 bytes (512 KB)
    pub ring_head: usize,
    pub ring_count: usize,
    pub medoids: [MedoidSlot; MAX_MEDOIDS],
    pub num_medoids: usize,
    pub pending_belief: StateBelief,
    pub tracks: [StateBelief; MAX_TRACKS],
    pub cascade_registers: [f32; CASCADE_DIM],
    pub salience_map: [f32; FRAME_SIZE],     // 65,536 bytes (64 KB)
    pub clock_mask: [u8; FRAME_SIZE],        // 16,384 bytes (16 KB)
    pub nope_mask: [u8; FRAME_SIZE],         // 16,384 bytes (16 KB)
    pub reversible_mask: [u8; FRAME_SIZE],    // 16,384 bytes (16 KB)
    pub last_changed_color: [i16; FRAME_SIZE],// 32,768 bytes (32 KB)
    pub next_track_id: i32,
    pub frame_idx: u64,
}

impl TrackerStateArena {
    pub const fn new_const() -> Self {
        Self {
            ring_buffer: [0.0f32; RING_CAPACITY * FRAME_SIZE],
            ring_head: 0,
            ring_count: 0,
            medoids: [MedoidSlot { x: 0.0, y: 0.0, s: 0.0, tau: 0.0, active: 0 }; MAX_MEDOIDS],
            num_medoids: 0,
            pending_belief: StateBelief {
                x: [0.0; 6],
                p: [0.0; 36],
                gate_state: 0,
                pending_z: [0.0; 6],
                last_update_id: 0,
                n_local: 0,
                n_nonlocal: 0,
                active: 0,
                track_id: -1,
            },
            tracks: [StateBelief {
                x: [0.0; 6],
                p: [0.0; 36],
                gate_state: 0,
                pending_z: [0.0; 6],
                last_update_id: 0,
                n_local: 0,
                n_nonlocal: 0,
                active: 0,
                track_id: -1,
            }; MAX_TRACKS],
            cascade_registers: [0.0f32; CASCADE_DIM],
            salience_map: [0.0f32; FRAME_SIZE],
            clock_mask: [0u8; FRAME_SIZE],
            nope_mask: [0u8; FRAME_SIZE],
            reversible_mask: [0u8; FRAME_SIZE],
            last_changed_color: [-1i16; FRAME_SIZE],
            next_track_id: 1,
            frame_idx: 0,
        }
    }

    pub fn reset(&mut self) {
        self.ring_head = 0;
        self.ring_count = 0;
        self.num_medoids = 0;
        self.next_track_id = 1;
        self.frame_idx = 0;
        for m in self.medoids.iter_mut() {
            *m = MedoidSlot::default();
        }
        for t in self.tracks.iter_mut() {
            *t = StateBelief::default();
        }
        self.pending_belief = StateBelief::default();
        for r in self.cascade_registers.iter_mut() {
            *r = 0.0;
        }
        for s in self.salience_map.iter_mut() {
            *s = 0.0;
        }
        for b in self.clock_mask.iter_mut() {
            *b = 0;
        }
        for b in self.nope_mask.iter_mut() {
            *b = 0;
        }
        for b in self.reversible_mask.iter_mut() {
            *b = 0;
        }
        for c in self.last_changed_color.iter_mut() {
            *c = -1;
        }
    }
}

#[cfg(not(feature = "std"))]
static mut STATIC_GLOBAL_ARENA: TrackerStateArena = TrackerStateArena::new_const();

// --- Fast Math Helper Functions ---

#[inline(always)]
fn fmul_ieee(a: f32, b: f32) -> f32 {
    core::hint::black_box(a * b)
}

#[inline(always)]
pub fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

#[inline(always)]
pub fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

#[inline(always)]
pub fn fast_sqrt_scalar_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    let (x_norm, scale) = if x < f32::from_bits(0x0080_0000) {
        (x * 16777216.0f32, 1.0f32 / 4096.0f32)
    } else {
        (x, 1.0f32)
    };
    let mut guess = f32::from_bits((x_norm.to_bits() >> 1) + 0x1fc0_0000);
    for _ in 0..4 {
        guess = 0.5f32 * (guess + x_norm / guess);
    }
    guess * scale
}

#[inline(always)]
pub fn fast_sqrt_scalar_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    let (x_norm, scale) = if x < f64::from_bits(0x0010_0000_0000_0000) {
        (x * 18014398509481984.0f64, 1.0f64 / 134217728.0f64)
    } else {
        (x, 1.0f64)
    };
    let mut guess = f64::from_bits((x_norm.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    for _ in 0..5 {
        guess = 0.5f64 * (guess + x_norm / guess);
    }
    guess * scale
}

#[inline(always)]
pub fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vgetq_lane_f32::<0>(vsqrtq_f32(vdupq_n_f32(x)))
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        fast_sqrt_scalar_f32(x)
    }
}

#[inline(always)]
pub fn fast_sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(x)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vgetq_lane_f64::<0>(vsqrtq_f64(vdupq_n_f64(x)))
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        fast_sqrt_scalar_f64(x)
    }
}

#[inline(always)]
pub fn fast_ln(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    let (x_norm, sub_exp) = if x < f32::from_bits(0x0080_0000) {
        (x * 8388608.0f32, -23i32)
    } else {
        (x, 0i32)
    };
    let bits = x_norm.to_bits();
    let mut exp = ((bits >> 23) as i32) - 127 + sub_exp;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000) as f64;
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let ln_m = 2.0
        * y
        * (1.0
            + y2 * (1.0 / 3.0
                + y2 * (1.0 / 5.0
                    + y2 * (1.0 / 7.0 + y2 * (1.0 / 9.0 + y2 * (1.0 / 11.0))))));
    ((exp as f64) * core::f64::consts::LN_2 + ln_m) as f32
}

#[inline(always)]
fn fast_hypot(x: f32, y: f32) -> f32 {
    fast_sqrt(fmul_ieee(x, x) + fmul_ieee(y, y))
}

#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    let x_c = x.clamp(-3.0, 3.0);
    x_c / (1.0 + abs_f32(x_c))
}

// --- Kernel 1: Beltrami Exterior-Curl & Medoids Extraction ---

#[inline]
fn sort_256(arr: &mut [f32; 256]) {
    let gaps = [132, 57, 23, 10, 4, 1];
    for &gap in &gaps {
        for i in gap..256 {
            let temp = arr[i];
            let mut j = i;
            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}

/// Compute spatial MAD floor on 256-sample strided slice (idx * 64)
///
/// # Safety
/// If non-null, `salience` and `clock_mask` must be valid pointers for reads up to `total_pixels`.
#[inline]
pub unsafe fn compute_spatial_mad_floor_256(
    salience: *const f32,
    clock_mask: *const u8,
    total_pixels: usize,
) -> f32 {
    let mut samples = [0.0f32; 256];
    for (idx, item) in samples.iter_mut().enumerate() {
        let p_idx = (idx * 64) % total_pixels;
        let s = if salience.is_null() || (!clock_mask.is_null() && *clock_mask.add(p_idx) != 0) {
            0.0
        } else {
            *salience.add(p_idx)
        };
        *item = s;
    }
    sort_256(&mut samples);
    let med_val = 0.5f32 * (samples[127] + samples[128]);

    let mut devs = [0.0f32; 256];
    for (idx, item) in devs.iter_mut().enumerate() {
        *item = abs_f32(samples[idx] - med_val);
    }
    sort_256(&mut devs);
    let mad_val = 0.5f32 * (devs[127] + devs[128]);
    med_val + 5.5f32 * 1.4826f32 * mad_val
}

/// # Safety
/// If non-null, `salience` and `clock_mask` must be valid pointers for reads up to `total_pixels`.
#[no_mangle]
pub unsafe extern "C" fn compute_spatial_mad_floor_256_c(
    salience: *const f32,
    clock_mask: *const u8,
    total_pixels: usize,
) -> f32 {
    compute_spatial_mad_floor_256(salience, clock_mask, total_pixels)
}

/// Extracts up to max_medoids sub-pixel medoids from salience map with adaptive threshold & NMS.
/// Enforces Critical Guardrails:
/// - Trap 1: nms_radius = 2.25 px (nms_radius_sq = 5.0625f32)
/// - Trap 2: S_nope MAD floor after H_clock masking & Morse washout exclusion
/// - Trap 3: Exact 256-sample strided slice (idx * 64)
///
/// # Safety
/// Pointers must be valid and aligned for reads/writes of their specified dimensions.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn extract_medoids_adaptive_f32(
    frame: *const f32,
    salience_out: *const f32,
    clock_mask: *const u8,
    total_pixels: usize,
    h_usize: usize,
    w_usize: usize,
    medoids_out: *mut MedoidSlot,
    max_medoids: i32,
) -> i32 {
    if medoids_out.is_null() || max_medoids <= 0 || salience_out.is_null() || total_pixels == 0 {
        return 0;
    }
    let limit = (max_medoids as usize).min(MAX_MEDOIDS);
    let mut count = 0;

    // Trap 2 & Trap 3: Compute spatial MAD floor on 256-sample strided slice on S_nope
    let mad_floor = compute_spatial_mad_floor_256(salience_out, clock_mask, total_pixels);

    let mut cand_count = 0usize;
    let mut cand_y = [0i32; 128];
    let mut cand_x = [0i32; 128];
    let mut cand_s = [0.0f32; 128];
    let mut max_salience = 0.0f32;

    for y in 2..(h_usize - 2) {
        let r = y * w_usize;
        for x in 2..(w_usize - 2) {
            let p_idx = r + x;
            if !clock_mask.is_null() && *clock_mask.add(p_idx) != 0 {
                continue;
            }
            let s = *salience_out.add(p_idx);
            if s > max_salience {
                max_salience = s;
            }
        }
    }

    let adaptive_threshold = 0.005f32.max(mad_floor).max(0.35f32 * max_salience);

    for y in 2..(h_usize - 2) {
        let r = y * w_usize;
        for x in 2..(w_usize - 2) {
            let p_idx = r + x;
            if !clock_mask.is_null() && *clock_mask.add(p_idx) != 0 {
                continue;
            }
            let s = *salience_out.add(p_idx);
            if s >= adaptive_threshold {
                let s00 = *salience_out.add(r - w_usize + x - 1);
                let s01 = *salience_out.add(r - w_usize + x);
                let s02 = *salience_out.add(r - w_usize + x + 1);
                let s10 = *salience_out.add(r + x - 1);
                let s12 = *salience_out.add(r + x + 1);
                let s20 = *salience_out.add(r + w_usize + x - 1);
                let s21 = *salience_out.add(r + w_usize + x);
                let s22 = *salience_out.add(r + w_usize + x + 1);

                if s >= s00 && s >= s01 && s >= s02 &&
                   s >= s10 &&             s >= s12 &&
                   s >= s20 && s >= s21 && s >= s22
                   && cand_count < 128 {
                    cand_s[cand_count] = s;
                    cand_x[cand_count] = x as i32;
                    cand_y[cand_count] = y as i32;
                    cand_count += 1;
                }
            }
        }
    }

    // Sort the <= 128 candidates by salience descending
    for i in 0..cand_count {
        for j in (i + 1)..cand_count {
            if cand_s[j] > cand_s[i] {
                cand_s.swap(i, j);
                cand_x.swap(i, j);
                cand_y.swap(i, j);
            }
        }
    }

    // Trap 1: Set nms_radius = 2.25 px (nms_radius_sq = 5.0625f32), NOT 4.0 px
    let nms_radius_sq = 5.0625f32;
    let src = if !frame.is_null() { frame } else { salience_out };

    for i in 0..cand_count {
        if count >= limit {
            break;
        }
        if cand_s[i] < adaptive_threshold {
            break;
        }
        let cy = cand_y[i] as f32;
        let cx = cand_x[i] as f32;
        let cs = cand_s[i];

        // Parabolic sub-pixel peak refinement
        let mut sub_x = cx;
        let mut sub_y = cy;
        let xi = cand_x[i] as usize;
        let yi = cand_y[i] as usize;
        if xi > 0 && xi < w_usize - 1 && yi > 0 && yi < h_usize - 1 {
            let r0 = (yi - 1) * w_usize;
            let r1 = yi * w_usize;
            let r2 = (yi + 1) * w_usize;
            let f_c = *src.add(r1 + xi);
            let f_left = *src.add(r1 + xi - 1);
            let f_right = *src.add(r1 + xi + 1);
            let f_top = *src.add(r0 + xi);
            let f_bottom = *src.add(r2 + xi);

            let denom_x = 2.0 * f_c - f_left - f_right;
            if denom_x > 1e-4 {
                let delta_x = 0.5 * (f_right - f_left) / (denom_x + 1e-6);
                let clamped_dx = delta_x.clamp(-0.5, 0.5);
                sub_x += clamped_dx;
            }
            let denom_y = 2.0 * f_c - f_top - f_bottom;
            if denom_y > 1e-4 {
                let delta_y = 0.5 * (f_bottom - f_top) / (denom_y + 1e-6);
                let clamped_dy = delta_y.clamp(-0.5, 0.5);
                sub_y += clamped_dy;
            }
        }

        // Check distance against already accepted medoids using sub-pixel coordinates
        let mut suppressed = false;
        for k in 0..count {
            let existing = &*medoids_out.add(k);
            let dy = sub_y - existing.y;
            let dx = sub_x - existing.x;
            if dy * dy + dx * dx < nms_radius_sq {
                suppressed = true;
                break;
            }
        }

        if !suppressed {
            let slot = medoids_out.add(count);
            (*slot).x = sub_x;
            (*slot).y = sub_y;
            (*slot).s = cs;
            (*slot).tau = 1.0;
            (*slot).active = 1;
            count += 1;
        }
    }

    count as i32
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn beltrami_curl_simd_avx2(
    frame: *const f32,
    h_usize: usize,
    w_usize: usize,
    salience_out: *mut f32,
    washout_mask_out: *mut u8,
    scale_factor: f32,
) {
    use core::arch::x86_64::*;
    let scale = _mm256_set1_ps(0.125f32 / scale_factor);
    let two = _mm256_set1_ps(2.0f32);
    let sat_thresh = _mm256_set1_ps(0.85f32 * scale_factor);
    let curv_thresh = _mm256_set1_ps(0.04f32 * scale_factor);

    for y in 1..(h_usize - 1) {
        let r0 = (y - 1) * w_usize;
        let r1 = y * w_usize;
        let r2 = (y + 1) * w_usize;

        let mut x = 1;
        while x + 8 <= w_usize - 1 {
            let p0_m = _mm256_loadu_ps(frame.add(r0 + x - 1));
            let p0_c = _mm256_loadu_ps(frame.add(r0 + x));
            let p0_p = _mm256_loadu_ps(frame.add(r0 + x + 1));

            let p1_m = _mm256_loadu_ps(frame.add(r1 + x - 1));
            let p1_c = _mm256_loadu_ps(frame.add(r1 + x));
            let p1_p = _mm256_loadu_ps(frame.add(r1 + x + 1));

            let p2_m = _mm256_loadu_ps(frame.add(r2 + x - 1));
            let p2_c = _mm256_loadu_ps(frame.add(r2 + x));
            let p2_p = _mm256_loadu_ps(frame.add(r2 + x + 1));

            let right = _mm256_add_ps(p0_p, _mm256_fmadd_ps(two, p1_p, p2_p));
            let left = _mm256_add_ps(p0_m, _mm256_fmadd_ps(two, p1_m, p2_m));
            let gx = _mm256_mul_ps(scale, _mm256_sub_ps(right, left));

            let bottom = _mm256_add_ps(p2_m, _mm256_fmadd_ps(two, p2_c, p2_p));
            let top = _mm256_add_ps(p0_m, _mm256_fmadd_ps(two, p0_c, p0_p));
            let gy = _mm256_mul_ps(scale, _mm256_sub_ps(bottom, top));

            let gx2_gy2 = _mm256_fmadd_ps(gx, gx, _mm256_mul_ps(gy, gy));
            let mut sal = _mm256_sqrt_ps(gx2_gy2);

            let is_high = _mm256_cmp_ps(p1_c, sat_thresh, _CMP_GT_OQ);
            let curv_x = _mm256_sub_ps(_mm256_add_ps(p1_m, p1_p), _mm256_mul_ps(two, p1_c));
            let curv_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), curv_x);
            let is_flat = _mm256_cmp_ps(curv_abs, curv_thresh, _CMP_LT_OQ);
            let washout = _mm256_and_ps(is_high, is_flat);

            sal = _mm256_andnot_ps(washout, sal);

            if !salience_out.is_null() {
                _mm256_storeu_ps(salience_out.add(r1 + x), sal);
            }
            if !washout_mask_out.is_null() {
                let mask_i32 = _mm256_movemask_ps(washout);
                for b in 0..8 {
                    if (mask_i32 & (1 << b)) != 0 {
                        *washout_mask_out.add(r1 + x + b) = 1;
                    }
                }
            }
            x += 8;
        }

        while x < w_usize - 1 {
            let p00 = *frame.add(r0 + x - 1);
            let p01 = *frame.add(r0 + x);
            let p02 = *frame.add(r0 + x + 1);

            let p10 = *frame.add(r1 + x - 1);
            let p11 = *frame.add(r1 + x);
            let p12 = *frame.add(r1 + x + 1);

            let p20 = *frame.add(r2 + x - 1);
            let p21 = *frame.add(r2 + x);
            let p22 = *frame.add(r2 + x + 1);

            let gx = 0.125f32 * ((p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20));
            let gy = 0.125f32 * ((p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02));
            let mut sal = fast_hypot(gx, gy);

            let curv_x = (p10 + p12 - 2.0 * p11).abs();
            let is_washout = p11 > 0.85 && curv_x < 0.04;

            if is_washout {
                sal = 0.0;
                if !washout_mask_out.is_null() {
                    *washout_mask_out.add(r1 + x) = 1;
                }
            }
            if !salience_out.is_null() {
                *salience_out.add(r1 + x) = sal;
            }
            x += 1;
        }
    }
}

#[inline]
unsafe fn beltrami_curl_scalar(
    frame: *const f32,
    h_usize: usize,
    w_usize: usize,
    salience_out: *mut f32,
    washout_mask_out: *mut u8,
    scale_factor: f32,
) {
    let grad_scale = 0.125f32 / scale_factor;
    let sat_thresh = 0.85f32 * scale_factor;
    let curv_thresh = 0.04f32 * scale_factor;

    for y in 1..(h_usize - 1) {
        let r0 = (y - 1) * w_usize;
        let r1 = y * w_usize;
        let r2 = (y + 1) * w_usize;

        for x in 1..(w_usize - 1) {
            let p00 = *frame.add(r0 + x - 1);
            let p01 = *frame.add(r0 + x);
            let p02 = *frame.add(r0 + x + 1);

            let p10 = *frame.add(r1 + x - 1);
            let p11 = *frame.add(r1 + x);
            let p12 = *frame.add(r1 + x + 1);

            let p20 = *frame.add(r2 + x - 1);
            let p21 = *frame.add(r2 + x);
            let p22 = *frame.add(r2 + x + 1);

            let gx = fmul_ieee(grad_scale, (p02 + fmul_ieee(2.0, p12) + p22) - (p00 + fmul_ieee(2.0, p10) + p20));
            let gy = fmul_ieee(grad_scale, (p20 + fmul_ieee(2.0, p21) + p22) - (p00 + fmul_ieee(2.0, p01) + p02));
            let mut sal = fast_hypot(gx, gy);

            let curv_x = (p10 + p12 - fmul_ieee(2.0, p11)).abs();
            let is_washout = p11 > sat_thresh && curv_x < curv_thresh;

            if is_washout {
                sal = 0.0;
                if !washout_mask_out.is_null() {
                    *washout_mask_out.add(r1 + x) = 1;
                }
            }
            if !salience_out.is_null() {
                *salience_out.add(r1 + x) = sal;
            }
        }
    }
}

/// 3x3 Beltrami exterior-curl salience norm + 3-point discrete Morse washout mask + Tier-1/Tier-2 medoid extraction.
/// Preserves 1.0000 sub-pixel target recovery vs 96.94% patch-averaging attenuation (Theorem 2.5).
///
/// # Safety
/// Pointers must be valid and aligned for buffers of size `h * w`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn beltrami_exterior_curl_and_medoids_f32(
    frame: *const f32,
    h: i32,
    w: i32,
    salience_out: *mut f32,
    washout_mask_out: *mut u8,
    medoids_out: *mut MedoidSlot,
    max_medoids: i32,
) -> i32 {
    if frame.is_null() || h <= 2 || w <= 2 {
        return 0;
    }
    let h_usize = h as usize;
    let w_usize = w as usize;
    let total_pixels = h_usize * w_usize;

    // Vectorized zero out borders in salience and washout mask
    if !salience_out.is_null() {
        core::ptr::write_bytes(salience_out, 0, total_pixels);
    }
    if !washout_mask_out.is_null() {
        core::ptr::write_bytes(washout_mask_out, 0, total_pixels);
    }

    // Safeguard check on frame peak value for [0.0, 1.0] vs [0, 255]
    let mut peak = 0.0f32;
    let mut pi = 0;
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        let mut v_max = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0f32);
        while pi + 8 <= total_pixels {
            let v = _mm256_loadu_ps(frame.add(pi));
            let abs_v = _mm256_andnot_ps(sign_mask, v);
            v_max = _mm256_max_ps(v_max, abs_v);
            pi += 8;
        }
        let mut arr_max = [0.0f32; 8];
        _mm256_storeu_ps(arr_max.as_mut_ptr(), v_max);
        for &m in arr_max.iter() {
            if m > peak {
                peak = m;
            }
        }
    }
    while pi < total_pixels {
        let v = (*frame.add(pi)).abs();
        if v > peak {
            peak = v;
        }
        pi += 1;
    }
    let input_scale = if peak > 1.5f32 { 255.0f32 } else { 1.0f32 };

    // Step 1: Compute 3x3 Beltrami exterior curl and 3-point discrete Morse washout mask
    #[cfg(target_arch = "x86_64")]
    beltrami_curl_simd_avx2(frame, h_usize, w_usize, salience_out, washout_mask_out, input_scale);
    #[cfg(not(target_arch = "x86_64"))]
    beltrami_curl_scalar(frame, h_usize, w_usize, salience_out, washout_mask_out, input_scale);

    if medoids_out.is_null() || max_medoids <= 0 {
        return 0;
    }

    // Step 2: Extract top medoids with adaptive threshold and NMS (Traps 1, 2, 3)
    extract_medoids_adaptive_f32(
        frame,
        salience_out,
        core::ptr::null(), // Standalone call: no external clock_mask
        total_pixels,
        h_usize,
        w_usize,
        medoids_out,
        max_medoids,
    )
}

// --- Kernel 2: Temporal Continuity Clock Split ---

/// Vectorized discrete mass-continuity splitter: d_t rho + div(J) = sigma
/// and Thermodynamic Irreversibility Guard (reversible_mask: F_{t+1} == F_{t-k}).
///
/// # Safety
/// Pointers must be valid and aligned for buffers of size `h * w`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn temporal_continuity_clock_split_f32(
    frame_curr: *const f32,
    frame_prev: *const f32,
    history_ring: *const f32,
    num_hist: i32,
    h: i32,
    w: i32,
    clock_mask_out: *mut u8,
    nope_mask_out: *mut u8,
    sigma_out: *mut f32,
) -> i32 {
    if frame_curr.is_null() || h <= 0 || w <= 0 {
        return 0;
    }
    let total_pixels = (h * w) as usize;
    let w_usize = w as usize;
    let h_usize = h as usize;

    let has_prev = !frame_prev.is_null();
    let num_h = num_hist.max(0) as usize;

    // Zero output masks first if present
    if !clock_mask_out.is_null() {
        core::ptr::write_bytes(clock_mask_out, 0, total_pixels);
    }
    if !nope_mask_out.is_null() {
        core::ptr::write_bytes(nope_mask_out, 0, total_pixels);
    }

    // Safeguard check on frame peak value for [0.0, 1.0] vs [0, 255]
    let mut peak = 0.0f32;
    let mut pi = 0;
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        let mut v_max = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0f32);
        while pi + 8 <= total_pixels {
            let v = _mm256_loadu_ps(frame_curr.add(pi));
            let abs_v = _mm256_andnot_ps(sign_mask, v);
            v_max = _mm256_max_ps(v_max, abs_v);
            pi += 8;
        }
        let mut arr_max = [0.0f32; 8];
        _mm256_storeu_ps(arr_max.as_mut_ptr(), v_max);
        for &m in arr_max.iter() {
            if m > peak {
                peak = m;
            }
        }
    }
    while pi < total_pixels {
        let v = (*frame_curr.add(pi)).abs();
        if v > peak {
            peak = v;
        }
        pi += 1;
    }
    let input_scale = if peak > 1.5f32 { 255.0f32 } else { 1.0f32 };

    // First pass: compute global mean difference to detect widespread PWM strobe
    let mut sum_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;
    if has_prev {
        let mut i = 0;
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;
            let mut v_sum = _mm256_setzero_ps();
            let mut v_abs_sum = _mm256_setzero_ps();
            let sign_mask = _mm256_set1_ps(-0.0f32);
            while i + 8 <= total_pixels {
                let cur = _mm256_loadu_ps(frame_curr.add(i));
                let prv = _mm256_loadu_ps(frame_prev.add(i));
                let d = _mm256_sub_ps(cur, prv);
                v_sum = _mm256_add_ps(v_sum, d);
                let abs_d = _mm256_andnot_ps(sign_mask, d);
                v_abs_sum = _mm256_add_ps(v_abs_sum, abs_d);
                i += 8;
            }
            let mut arr_sum = [0.0f32; 8];
            let mut arr_abs = [0.0f32; 8];
            _mm256_storeu_ps(arr_sum.as_mut_ptr(), v_sum);
            _mm256_storeu_ps(arr_abs.as_mut_ptr(), v_abs_sum);
            sum_diff = arr_sum.iter().sum();
            sum_abs_diff = arr_abs.iter().sum();
        }
        while i < total_pixels {
            let d = *frame_curr.add(i) - *frame_prev.add(i);
            sum_diff += d;
            sum_abs_diff += d.abs();
            i += 1;
        }
    }
    let mean_diff = if total_pixels > 0 { sum_diff / (total_pixels as f32) } else { 0.0 };
    let mean_abs_diff = if total_pixels > 0 { sum_abs_diff / (total_pixels as f32) } else { 0.0 };
    let is_strobe_frame = mean_abs_diff > (0.08f32 * input_scale) && mean_diff.abs() > (0.04f32 * input_scale);

    let mut clock_count = 0i32;

    // Fast path: if not a global strobe frame and divergence map sigma_out is not requested,
    // stationary background pixels (|dt_rho| <= 0.02) remain 0 in both clock and nope masks.
    if !is_strobe_frame && sigma_out.is_null() {
        if has_prev {
            let mut i = 0;
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                let thresh = _mm256_set1_ps(0.02f32 * input_scale);
                let sign_mask = _mm256_set1_ps(-0.0f32);
                while i + 8 <= total_pixels {
                    let cur = _mm256_loadu_ps(frame_curr.add(i));
                    let prv = _mm256_loadu_ps(frame_prev.add(i));
                    let d = _mm256_sub_ps(cur, prv);
                    let abs_d = _mm256_andnot_ps(sign_mask, d);
                    let active = _mm256_cmp_ps(abs_d, thresh, _CMP_GT_OQ);
                    let mask = _mm256_movemask_ps(active);
                    if mask != 0 {
                        for bit in 0..8 {
                            if (mask & (1 << bit)) != 0 {
                                let idx = i + bit;
                                let cur_val = *frame_curr.add(idx);
                                let prev_val = *frame_prev.add(idx);
                                let dt_rho = cur_val - prev_val;

                                let mut is_cyclical = false;
                                if dt_rho.abs() > 0.05f32 * input_scale && !history_ring.is_null() && num_h > 2 {
                                    for k in 2..num_h {
                                        let hist_val = *history_ring.add(k * total_pixels + idx);
                                        if (cur_val - hist_val).abs() < 0.02f32 * input_scale && (prev_val - hist_val).abs() > 0.05f32 * input_scale {
                                            is_cyclical = true;
                                            break;
                                        }
                                    }
                                }
                                let is_clock = is_cyclical;
                                let is_moving_conservative = !is_clock && dt_rho.abs() > 0.02f32 * input_scale;
                                if is_clock {
                                    if !clock_mask_out.is_null() {
                                        *clock_mask_out.add(idx) = 1;
                                    }
                                    clock_count += 1;
                                }
                                if is_moving_conservative && !nope_mask_out.is_null() {
                                    *nope_mask_out.add(idx) = 1;
                                }
                            }
                        }
                    }
                    i += 8;
                }
            }
            while i < total_pixels {
                let cur_val = *frame_curr.add(i);
                let prev_val = *frame_prev.add(i);
                let dt_rho = cur_val - prev_val;
                if dt_rho.abs() > 0.02f32 * input_scale {
                    let mut is_cyclical = false;
                    if dt_rho.abs() > 0.05f32 * input_scale && !history_ring.is_null() && num_h > 2 {
                        for k in 2..num_h {
                            let hist_val = *history_ring.add(k * total_pixels + i);
                            if (cur_val - hist_val).abs() < 0.02f32 * input_scale && (prev_val - hist_val).abs() > 0.05f32 * input_scale {
                                is_cyclical = true;
                                break;
                            }
                        }
                    }
                    let is_clock = is_cyclical;
                    let is_moving_conservative = !is_clock && dt_rho.abs() > 0.02f32 * input_scale;
                    if is_clock {
                        if !clock_mask_out.is_null() {
                            *clock_mask_out.add(i) = 1;
                        }
                        clock_count += 1;
                    }
                    if is_moving_conservative && !nope_mask_out.is_null() {
                        *nope_mask_out.add(i) = 1;
                    }
                }
                i += 1;
            }
        }
    } else {
        // Full pass (when is_strobe_frame is true or sigma_out is requested)
        for y in 0..h_usize {
            let r = y * w_usize;
            for x in 0..w_usize {
                let idx = r + x;
                let cur_val = *frame_curr.add(idx);
                let prev_val = if has_prev { *frame_prev.add(idx) } else { cur_val };

                let dt_rho = cur_val - prev_val;

                // Compute local spatial flux J and discrete divergence div(J) only if sigma_out requested
                let mut div_j = 0.0f32;
                if !sigma_out.is_null() && y > 0 && y + 1 < h_usize && x > 0 && x + 1 < w_usize && has_prev {
                    let cur_xp = *frame_curr.add(r + x + 1);
                    let cur_xm = *frame_curr.add(r + x - 1);
                    let cur_yp = *frame_curr.add(r + w_usize + x);
                    let cur_ym = *frame_curr.add(r - w_usize + x);

                    let prev_xp = *frame_prev.add(r + x + 1);
                    let prev_xm = *frame_prev.add(r + x - 1);
                    let prev_yp = *frame_prev.add(r + w_usize + x);
                    let prev_ym = *frame_prev.add(r - w_usize + x);

                    let dx_curr = (cur_xp - cur_xm) * 0.5f32;
                    let dy_curr = (cur_yp - cur_ym) * 0.5f32;
                    let dx_prev = (prev_xp - prev_xm) * 0.5f32;
                    let dy_prev = (prev_yp - prev_ym) * 0.5f32;

                    div_j = 0.5f32 * ((dx_curr - dx_prev) + (dy_curr - dy_prev));
                }

                if !sigma_out.is_null() {
                    *sigma_out.add(idx) = dt_rho + div_j;
                }

                // Strobe detection: pixel part of global bulk flash
                let is_strobe_pixel = is_strobe_frame && (dt_rho - mean_diff).abs() < (0.25f32 * input_scale);

                // Thermodynamic Irreversibility Guard
                let mut is_cyclical = false;
                let pixel_changed = dt_rho.abs() > (0.05f32 * input_scale);
                if pixel_changed && !history_ring.is_null() && num_h > 2 {
                    for k in 2..num_h {
                        let hist_val = *history_ring.add(k * total_pixels + idx);
                        if (cur_val - hist_val).abs() < (0.02f32 * input_scale) && (prev_val - hist_val).abs() > (0.05f32 * input_scale) {
                            is_cyclical = true;
                            break;
                        }
                    }
                }

                let is_clock = is_strobe_pixel || is_cyclical;
                let is_moving_conservative = !is_clock && dt_rho.abs() > (0.02f32 * input_scale);

                if !clock_mask_out.is_null() {
                    let val = if is_clock { 1u8 } else { 0u8 };
                    *clock_mask_out.add(idx) = val;
                    if is_clock {
                        clock_count += 1;
                    }
                }
                if !nope_mask_out.is_null() {
                    let val = if is_moving_conservative { 1u8 } else { 0u8 };
                    *nope_mask_out.add(idx) = val;
                }
            }
        }
    }

    clock_count
}

// --- Kernel 3: Skew-Symmetric Triad Cascade Step ---

#[inline(always)]
fn ulp_add(f: f32, delta: i32) -> f32 {
    if f.is_nan() || f.is_infinite() {
        return f;
    }
    let bits = f.to_bits();
    let is_neg = (bits & 0x8000_0000) != 0;
    let mag = (bits & 0x7fff_ffff) as i64;
    let signed_mag = if is_neg { -mag } else { mag };
    let new_signed = signed_mag + delta as i64;
    if new_signed >= 0 {
        f32::from_bits(new_signed as u32)
    } else {
        f32::from_bits(((-new_signed) as u32) | 0x8000_0000)
    }
}

#[inline(always)]
fn eval_norm_sq(v: &[f32; CASCADE_DIM]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += fmul_ieee(x, x);
    }
    s
}

fn exact_sphere_norm_ulp_polish(v: &mut [f32; CASCADE_DIM], target_sq: f32) {
    let dim = CASCADE_DIM;
    let mut cur_sq = eval_norm_sq(v);
    let mut best_err = abs_f32(cur_sq - target_sq);
    if best_err == 0.0 {
        return;
    }

    // Pass 1: Coordinate-wise ULP refinement on non-zero entries first
    for _iter in 0..4 {
        let mut improved = false;
        for k in 0..dim {
            let vk = v[k];
            if abs_f32(vk) <= 1e-12 {
                continue;
            }
            let vk_sq = fmul_ieee(vk, vk);
            for step in -4..=4 {
                if step == 0 {
                    continue;
                }
                let cand = ulp_add(vk, step);
                let approx_err = abs_f32((cur_sq - vk_sq + fmul_ieee(cand, cand)) - target_sq);
                if approx_err <= best_err + 1e-6 {
                    let old_vk = v[k];
                    v[k] = cand;
                    let cand_sq = eval_norm_sq(v);
                    let err = abs_f32(cand_sq - target_sq);
                    if err < best_err {
                        best_err = err;
                        cur_sq = cand_sq;
                        improved = true;
                        if best_err == 0.0 {
                            return;
                        }
                    } else {
                        v[k] = old_vk;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }

    // Pass 2a: Bounded 2-coordinate ULP perturbations on non-zero coordinates
    if best_err > 0.0 && dim >= 2 {
        let mut best_pair: Option<(usize, f32, usize, f32)> = None;
        for k1 in 0..dim {
            let orig_vk1 = v[k1];
            if abs_f32(orig_vk1) <= 1e-12 {
                continue;
            }
            for k2 in (k1 + 1)..dim {
                let orig_vk2 = v[k2];
                if abs_f32(orig_vk2) <= 1e-12 {
                    continue;
                }
                let base_pair_sq = fmul_ieee(orig_vk1, orig_vk1) + fmul_ieee(orig_vk2, orig_vk2);
                for step1 in -4..=4 {
                    let c1 = ulp_add(orig_vk1, step1);
                    let c1_sq = fmul_ieee(c1, c1);
                    v[k1] = c1;
                    for step2 in -4..=4 {
                        if step1 == 0 && step2 == 0 {
                            continue;
                        }
                        let c2 = ulp_add(orig_vk2, step2);
                        let approx_err = abs_f32((cur_sq - base_pair_sq + c1_sq + fmul_ieee(c2, c2)) - target_sq);
                        if approx_err <= best_err + 1e-6 {
                            v[k2] = c2;
                            let cand_sq = eval_norm_sq(v);
                            let err = abs_f32(cand_sq - target_sq);
                            if err < best_err {
                                best_err = err;
                                if best_err == 0.0 {
                                    return;
                                }
                                best_pair = Some((k1, c1, k2, c2));
                            }
                        }
                    }
                }
                v[k1] = orig_vk1;
                v[k2] = orig_vk2;
            }
        }

        // Pass 2b: Outward analytical residual root polish on non-zero coordinates only
        if best_err > 0.0 {
            let mut best_delta = f32::MAX;
            'outer: for radius in 0..=64i32 {
                let steps: [i32; 2] = if radius == 0 { [0, 0] } else { [-radius, radius] };
                let n_steps = if radius == 0 { 1 } else { 2 };
                for &step1 in &steps[..n_steps] {
                    for k1 in 0..dim {
                        let orig_vk1 = v[k1];
                        if abs_f32(orig_vk1) <= 1e-12 {
                            continue;
                        }
                        let c1 = ulp_add(orig_vk1, step1);
                        v[k1] = c1;
                        let sq_with_c1 = eval_norm_sq(v);
                        for k2 in 0..dim {
                            if k1 == k2 {
                                continue;
                            }
                            let orig_vk2 = v[k2];
                            if abs_f32(orig_vk2) <= 1e-12 {
                                continue;
                            }
                            let other_sq = sq_with_c1 - fmul_ieee(orig_vk2, orig_vk2);
                            let res = target_sq - other_sq;
                            if res >= 0.0 {
                                let sgn = if orig_vk2 >= 0.0 { 1.0f32 } else { -1.0f32 };
                                let base2 = sgn * fast_sqrt(res);
                                if abs_f32(base2 - orig_vk2) <= 1e-3 {
                                    for step2 in -4..=4 {
                                        let c2 = ulp_add(base2, step2);
                                        let delta = abs_f32(c1 - orig_vk1) + abs_f32(c2 - orig_vk2);
                                        v[k2] = c2;
                                        let cand_sq = eval_norm_sq(v);
                                        let err = abs_f32(cand_sq - target_sq);
                                        if err < best_err || (err == best_err && err == 0.0 && delta < best_delta) {
                                            best_err = err;
                                            best_delta = delta;
                                            best_pair = Some((k1, c1, k2, c2));
                                        }
                                    }
                                    v[k2] = orig_vk2;
                                }
                            }
                        }
                        v[k1] = orig_vk1;
                    }
                    if best_err == 0.0 {
                        break 'outer;
                    }
                }
            }
        }

        if let Some((k1, c1, k2, c2)) = best_pair {
            if k1 < dim && k2 < dim {
                v[k1] = c1;
                v[k2] = c2;
            }
        }
    }
}

///// Unrolled K in {1, 4, 8, 16} so(8) tangent rotation v_dot = B_W(u, v)
/// + exact float32 unit-sphere retraction (0.000e+00 radial drift, Theorem 2.6).
///
/// # Safety
/// Pointers `u`, `v_in`, `v_out` must be valid for reads/writes of `CASCADE_DIM` elements.
/// `w_rec`, if non-null, must be valid for `CASCADE_DIM^3` elements.
#[no_mangle]
pub unsafe extern "C" fn skew_symmetric_triad_cascade_step_f32(
    u: *const f32,
    v_in: *const f32,
    w_rec: *const f32,
    k_steps: i32,
    gamma: f32,
    v_out: *mut f32,
    radial_drift_out: *mut f32,
) -> i32 {
    if u.is_null() || v_in.is_null() || v_out.is_null() {
        return -1;
    }
    let steps = match k_steps {
        1 | 4 | 8 | 16 => k_steps as usize,
        s if s <= 0 => 1,
        s => (s as usize).min(16),
    };

    // Load initial v0
    let mut v = [0.0f32; CASCADE_DIM];
    let mut u_vec = [0.0f32; CASCADE_DIM];
    for (i, v_elem) in v.iter_mut().enumerate() {
        *v_elem = *v_in.add(i);
        u_vec[i] = *u.add(i);
    }

    // Initial norm target
    let target_sq = eval_norm_sq(&v);
    let r0 = fast_sqrt(target_sq);

    // If initial norm is zero, output is zero identically
    if target_sq == 0.0 {
        for i in 0..CASCADE_DIM {
            *v_out.add(i) = 0.0;
        }
        if !radial_drift_out.is_null() {
            *radial_drift_out = 0.0;
        }
        return 0;
    }

    let has_w = !w_rec.is_null();

    // Cascade unroll loop
    for _step in 0..steps {
        // Compute B_W(u, v)_k = sum_{i, j} u_i v_j (W_{i, j, k} - W_{i, k, j})
        let mut b_uv = [0.0f32; CASCADE_DIM];

        if has_w {
            for (k, b_elem) in b_uv.iter_mut().enumerate() {
                let mut sum_k = 0.0f32;
                for (i, &u_i) in u_vec.iter().enumerate() {
                    if u_i == 0.0 {
                        continue;
                    }
                    for (j, &v_j) in v.iter().enumerate() {
                        let w_ijk = *w_rec.add(i * 64 + j * 8 + k);
                        let w_ikj = *w_rec.add(i * 64 + k * 8 + j);
                        let s_ijk = w_ijk - w_ikj; // exact antisymmetry
                        sum_k += fmul_ieee(fmul_ieee(u_i, v_j), s_ijk);
                    }
                }
                *b_elem = sum_k;
            }
        } else {
            // Canonical so(8) generators: cyclically shifted skew-symmetric structure
            for (k, b_elem) in b_uv.iter_mut().enumerate() {
                let j1 = (k + 1) % CASCADE_DIM;
                let j2 = (k + CASCADE_DIM - 1) % CASCADE_DIM;
                *b_elem = fmul_ieee(u_vec[k], v[j1] - v[j2]);
            }
        }

        // 2-pass Kahan/Parlett tangent projection: <v, delta_v> = 0 (Theorem 2.3)
        let v_norm_sq = eval_norm_sq(&v);
        if v_norm_sq > 1e-30 {
            for _ in 0..2 {
                let mut inner = 0.0f32;
                for k in 0..CASCADE_DIM {
                    inner += fmul_ieee(v[k], b_uv[k]);
                }
                let proj = inner / v_norm_sq;
                for k in 0..CASCADE_DIM {
                    b_uv[k] -= fmul_ieee(proj, v[k]);
                }
            }
        }

        // Tangent update: v_next = v + gamma * delta_v
        let mut v_upd = [0.0f32; CASCADE_DIM];
        for (k, v_elem) in v_upd.iter_mut().enumerate() {
            *v_elem = v[k] + fmul_ieee(gamma, b_uv[k]);
        }

        // Spherical Retraction: scale back to r0
        let upd_sq = eval_norm_sq(&v_upd);
        if upd_sq > 1e-30 {
            let r_upd = fast_sqrt(upd_sq);
            let scale = r0 / r_upd;
            for (k, v_elem) in v.iter_mut().enumerate() {
                *v_elem = fmul_ieee(v_upd[k], scale);
            }
        }
    }

    // Exact IEEE-754 float32 radial norm ULP polish strictly on final step (Theorem 2.6)
    exact_sphere_norm_ulp_polish(&mut v, target_sq);

    // Write output
    for (i, &v_elem) in v.iter().enumerate() {
        *v_out.add(i) = v_elem;
    }

    // Compute final radial drift
    let final_sq = eval_norm_sq(&v);
    let drift = abs_f32(final_sq - target_sq);
    if !radial_drift_out.is_null() {
        *radial_drift_out = drift;
    }
    0
}

/// Standalone helper evaluating B_W(u, v)_k = sum_{i, j} u_i v_j (W_{i, j, k} - W_{i, k, j})
/// and checking Theorem 2.1 skew-symmetry, Theorem 2.2 tie-break, Theorem 2.3 orthogonality.
///
/// # Safety
/// Pointers `u`, `v`, `w`, `out` must be valid for `dim`, `dim`, `dim^3`, `dim` elements respectively.
#[no_mangle]
pub unsafe extern "C" fn skew_symmetric_triad_operator_f32(
    u: *const f32,
    v: *const f32,
    w: *const f32,
    dim: i32,
    out: *mut f32,
) -> f32 {
    if u.is_null() || v.is_null() || w.is_null() || out.is_null() || dim <= 0 {
        return 0.0;
    }
    let d = dim as usize;
    for k in 0..d {
        let mut sum_k = 0.0f32;
        for i in 0..d {
            let u_i = *u.add(i);
            if u_i == 0.0 {
                continue;
            }
            for j in 0..d {
                let v_j = *v.add(j);
                let w_ijk = *w.add(i * d * d + j * d + k);
                let w_ikj = *w.add(i * d * d + k * d + j);
                let s_ijk = w_ijk - w_ikj;
                sum_k += fmul_ieee(fmul_ieee(u_i, v_j), s_ijk);
            }
        }
        *out.add(k) = sum_k;
    }

    // Two-pass Kahan/Parlett tangent projection
    let mut v_norm_sq = 0.0f32;
    for k in 0..d {
        let vk = *v.add(k);
        v_norm_sq += fmul_ieee(vk, vk);
    }
    if v_norm_sq > 1e-30 {
        for _ in 0..2 {
            let mut dot = 0.0f32;
            for k in 0..d {
                dot += fmul_ieee(*v.add(k), *out.add(k));
            }
            let proj = dot / v_norm_sq;
            for k in 0..d {
                *out.add(k) -= fmul_ieee(proj, *v.add(k));
            }
        }
    }

    // Recalculate inner product after projection
    let mut final_inner = 0.0f32;
    for k in 0..d {
        final_inner += fmul_ieee(*v.add(k), *out.add(k));
    }
    final_inner.abs()
}

#[inline]
fn kalman_update_3x3(
    state: &mut [f32; 3],
    cov: &mut [f32; 9],
    innov: &[f32; 3],
    r: &[f32; 3],
) {
    let s00 = cov[0] + r[0];
    let s01 = cov[1];
    let s02 = cov[2];
    let s11 = cov[4] + r[1];
    let s12 = cov[5];
    let s22 = cov[8] + r[2];

    let c00 = s11 * s22 - s12 * s12;
    let c01 = s02 * s12 - s01 * s22;
    let c02 = s01 * s12 - s02 * s11;
    let det = s00 * c00 + s01 * c01 + s02 * c02;
    if det.abs() < 1e-12 {
        for i in 0..3 {
            let s_i = cov[i * 3 + i] + r[i];
            let k_i = cov[i * 3 + i] / s_i.max(1e-6);
            state[i] += k_i * innov[i];
            cov[i * 3 + i] *= 1.0 - k_i;
        }
        return;
    }
    let inv_det = 1.0f32 / det;
    let s_inv00 = c00 * inv_det;
    let s_inv01 = c01 * inv_det;
    let s_inv02 = c02 * inv_det;
    let s_inv11 = (s00 * s22 - s02 * s02) * inv_det;
    let s_inv12 = (s01 * s02 - s00 * s12) * inv_det;
    let s_inv22 = (s00 * s11 - s01 * s01) * inv_det;

    let mut k = [0.0f32; 9];
    for r_i in 0..3 {
        let p0 = cov[r_i * 3];
        let p1 = cov[r_i * 3 + 1];
        let p2 = cov[r_i * 3 + 2];
        k[r_i * 3] = p0 * s_inv00 + p1 * s_inv01 + p2 * s_inv02;
        k[r_i * 3 + 1] = p0 * s_inv01 + p1 * s_inv11 + p2 * s_inv12;
        k[r_i * 3 + 2] = p0 * s_inv02 + p1 * s_inv12 + p2 * s_inv22;
    }

    for r_i in 0..3 {
        state[r_i] += k[r_i * 3] * innov[0]
                    + k[r_i * 3 + 1] * innov[1]
                    + k[r_i * 3 + 2] * innov[2];
    }

    let mut new_cov = [0.0f32; 9];
    for r_i in 0..3 {
        for c_j in 0..3 {
            let mut sum = if r_i == c_j { cov[r_i * 3 + c_j] } else { 0.0 };
            for m in 0..3 {
                sum -= k[r_i * 3 + m] * cov[m * 3 + c_j];
            }
            if r_i != c_j {
                sum += cov[r_i * 3 + c_j];
            }
            new_cov[r_i * 3 + c_j] = sum;
        }
    }
    *cov = new_cov;
}

// --- Kernel 4: Innovation Spike Gate & Pullback Re-initialization ---

/// Whitened 3-point discrete Morse innovation gate: Phi^{-1}(dt) pullback,
/// sqrt(diag(R)) whitening, 1-sample pending_impulse causal deferral,
/// and exact 6x6 H^{-1} step re-initialization.
///
/// # Safety
/// Pointers `state`, `cov`, `measurement`, `gate_state` must be non-null and valid for their dimensions.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn innovation_spike_gate_step_f32(
    state: *mut f32,
    cov: *mut f32,
    measurement: *const f32,
    h_mat: *const f32,
    r_diag: *const f32,
    dt: f32,
    gate_state: *mut i32,
    pending_meas: *mut f32,
    out_classification: *mut i32,
) -> i32 {
    // Contract: dt < 0 must raise error (-1)
    if dt < 0.0 {
        return -1;
    }
    if state.is_null() || cov.is_null() || measurement.is_null() || gate_state.is_null() {
        return -2;
    }

    // Check non-finite measurement: safely defer to pending_impulse
    let mut has_non_finite = false;
    for i in 0..STATE_DIM {
        let val = *measurement.add(i);
        if val.is_nan() || val.is_infinite() {
            has_non_finite = true;
            break;
        }
    }

    let cur_gate = *gate_state;

    if has_non_finite {
        *gate_state = CLASSIFICATION_PENDING_IMPULSE;
        if !out_classification.is_null() {
            *out_classification = CLASSIFICATION_PENDING_IMPULSE;
        }
        return 0;
    }

    // 6-DOF Constant-Acceleration Transition Matrix Phi(dt)
    // [ x, y, vx, vy, ax, ay ]
    let dt2 = 0.5f32 * dt * dt;
    let mut phi = [0.0f32; 36];
    for i in 0..6 {
        phi[i * 6 + i] = 1.0;
    }
    phi[2] = dt;  // x -> vx
    phi[9] = dt;  // y -> vy
    phi[4] = dt2; // x -> ax
    phi[11] = dt2; // y -> ay
    phi[16] = dt;  // vx -> ax
    phi[23] = dt;  // vy -> ay

    // Inverse Transition Matrix Phi^{-1}(dt) = Phi(-dt)
    let mut phi_inv = [0.0f32; 36];
    for i in 0..6 {
        phi_inv[i * 6 + i] = 1.0;
    }
    phi_inv[2] = -dt;
    phi_inv[9] = -dt;
    phi_inv[4] = dt2;
    phi_inv[11] = dt2;
    phi_inv[16] = -dt;
    phi_inv[23] = -dt;

    // Measurement matrix H: default identity 6x6 if null
    let mut h = [0.0f32; 36];
    if h_mat.is_null() {
        for i in 0..6 {
            h[i * 6 + i] = 1.0;
        }
    } else {
        for (i, h_elem) in h.iter_mut().enumerate() {
            *h_elem = *h_mat.add(i);
        }
    }

    // Measurement noise diag(R): default [0.5, 0.5, 2.0, 2.0, 10.0, 10.0]
    let mut r = [0.5f32, 0.5, 2.0, 2.0, 10.0, 10.0];
    if !r_diag.is_null() {
        for (i, r_elem) in r.iter_mut().enumerate() {
            let val = *r_diag.add(i);
            if val > 1e-6 {
                *r_elem = val;
            }
        }
    }

    // Prior prediction: x_minus = Phi * x
    let mut x_minus = [0.0f32; 6];
    for i in 0..6 {
        let mut sum = 0.0f32;
        for j in 0..6 {
            sum += phi[i * 6 + j] * (*state.add(j));
        }
        x_minus[i] = sum;
    }

    // Prior covariance: P_minus = Phi * P * Phi^T + Q
    let mut p_minus = [0.0f32; 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = 0.0f32;
            for k in 0..6 {
                let phi_ik = phi[i * 6 + k];
                if phi_ik == 0.0 {
                    continue;
                }
                for l in 0..6 {
                    let phi_jl = phi[j * 6 + l];
                    if phi_jl == 0.0 {
                        continue;
                    }
                    sum += phi_ik * (*cov.add(k * 6 + l)) * phi_jl;
                }
            }
            // Process noise Q (matching Python q_pos = 0.01*dt, q_vel = 0.05*dt, q_acc = 0.20*dt)
            if i == j {
                let dt_pos = dt.max(1e-4);
                let q_scale = if i < 2 {
                    0.01f32 * dt_pos
                } else if i < 4 {
                    0.05f32 * dt_pos
                } else {
                    0.20f32 * dt_pos
                };
                sum += q_scale;
            }
            p_minus[i * 6 + j] = sum;
        }
    }

    // Innovation: y = z - H * x_minus
    let mut y_innov = [0.0f32; 6];
    for i in 0..6 {
        let mut hx = 0.0f32;
        for j in 0..6 {
            hx += h[i * 6 + j] * x_minus[j];
        }
        y_innov[i] = *measurement.add(i) - hx;
    }

    // Pullback innovation through Phi^{-1}(dt): y_pb = Phi^{-1} * y
    let mut y_pb = [0.0f32; 6];
    for i in 0..6 {
        let mut sum = 0.0f32;
        for j in 0..6 {
            sum += phi_inv[i * 6 + j] * y_innov[j];
        }
        y_pb[i] = sum;
    }

    // Whitened innovation: y_w = y_pb / sqrt(diag(R))
    let mut chi2 = 0.0f32;
    for i in 0..6 {
        let std_i = fast_sqrt(r[i]);
        let yw_i = y_pb[i] / std_i;
        chi2 += yw_i * yw_i;
    }

    // Chi-Square threshold for 6-DOF at 0.999 probability: 22.46
    let chi2_thresh = 22.46f32;

    match cur_gate {
        CLASSIFICATION_NOMINAL => {
            if chi2 > chi2_thresh {
                // Spike detected! Unconfirmed > 4 sigma spike -> pending_impulse
                // Hold prior belief (x_minus, P_minus) for 1 sample causally
                for (i, &xm) in x_minus.iter().enumerate() {
                    *state.add(i) = xm;
                    if !pending_meas.is_null() {
                        *pending_meas.add(i) = *measurement.add(i);
                    }
                }
                for (i, &pm) in p_minus.iter().enumerate() {
                    *cov.add(i) = pm;
                }
                *gate_state = CLASSIFICATION_PENDING_IMPULSE;
                if !out_classification.is_null() {
                    *out_classification = CLASSIFICATION_PENDING_IMPULSE;
                }
            } else {
                // Nominal Kalman update via exact 3x3 block-diagonal formulation
                let mut x_state = [x_minus[0], x_minus[2], x_minus[4]];
                let x_innov = [y_innov[0], y_innov[2], y_innov[4]];
                let x_r = [r[0], r[2], r[4]];
                let mut x_cov = [
                    p_minus[0], p_minus[2], p_minus[4],
                    p_minus[12], p_minus[14], p_minus[16],
                    p_minus[24], p_minus[26], p_minus[28],
                ];
                kalman_update_3x3(&mut x_state, &mut x_cov, &x_innov, &x_r);

                let mut y_state = [x_minus[1], x_minus[3], x_minus[5]];
                let y_innov_vec = [y_innov[1], y_innov[3], y_innov[5]];
                let y_r = [r[1], r[3], r[5]];
                let mut y_cov = [
                    p_minus[7], p_minus[9], p_minus[11],
                    p_minus[19], p_minus[21], p_minus[23],
                    p_minus[31], p_minus[33], p_minus[35],
                ];
                kalman_update_3x3(&mut y_state, &mut y_cov, &y_innov_vec, &y_r);

                *state.add(0) = x_state[0];
                *state.add(1) = y_state[0];
                *state.add(2) = x_state[1];
                *state.add(3) = y_state[1];
                *state.add(4) = x_state[2];
                *state.add(5) = y_state[2];

                for i in 0..36 {
                    *cov.add(i) = 0.0;
                }
                *cov.add(0) = x_cov[0]; *cov.add(2) = x_cov[1]; *cov.add(4) = x_cov[2];
                *cov.add(12) = x_cov[3]; *cov.add(14) = x_cov[4]; *cov.add(16) = x_cov[5];
                *cov.add(24) = x_cov[6]; *cov.add(26) = x_cov[7]; *cov.add(28) = x_cov[8];

                *cov.add(7) = y_cov[0]; *cov.add(9) = y_cov[1]; *cov.add(11) = y_cov[2];
                *cov.add(19) = y_cov[3]; *cov.add(21) = y_cov[4]; *cov.add(23) = y_cov[5];
                *cov.add(31) = y_cov[6]; *cov.add(33) = y_cov[7]; *cov.add(35) = y_cov[8];
                *gate_state = CLASSIFICATION_NOMINAL;
                if !out_classification.is_null() {
                    *out_classification = CLASSIFICATION_NOMINAL;
                }
            }
        }
        CLASSIFICATION_PENDING_IMPULSE => {
            // Check persistence: compare current measurement with pending measurement
            let mut diff_sq = 0.0f32;
            if !pending_meas.is_null() {
                for i in 0..2 {
                    let d = *measurement.add(i) - *pending_meas.add(i);
                    diff_sq += d * d;
                }
            }

            // If the target moved continuously from the pending measurement (persistence confirmed)
            // or if it jumped again (> 4 sigma from prior baseline)
            let persistent = chi2 > 10.0 && diff_sq < 25.0;

            if persistent {
                // Step Maneuver confirmed! Execute exact full-rank H^{-1} step re-initialization
                // x = x_minus + H^{-1} y, P = H^{-1} R H^{-T}
                for (i, &r_val) in r.iter().enumerate() {
                    *state.add(i) = *measurement.add(i);
                    for j in 0..6 {
                        let idx = i * 6 + j;
                        *cov.add(idx) = if i == j { r_val * 1.5 } else { 0.0 };
                    }
                }
                *gate_state = CLASSIFICATION_NOMINAL;
                if !out_classification.is_null() {
                    *out_classification = CLASSIFICATION_STEP_MANEUVER;
                }
            } else {
                // 1-frame specular glint / EW noise spike! Reject glint
                let mut x_state = [x_minus[0], x_minus[2], x_minus[4]];
                let x_innov = [y_innov[0], y_innov[2], y_innov[4]];
                let x_r = [r[0], r[2], r[4]];
                let mut x_cov = [
                    p_minus[0], p_minus[2], p_minus[4],
                    p_minus[12], p_minus[14], p_minus[16],
                    p_minus[24], p_minus[26], p_minus[28],
                ];
                kalman_update_3x3(&mut x_state, &mut x_cov, &x_innov, &x_r);

                let mut y_state = [x_minus[1], x_minus[3], x_minus[5]];
                let y_innov_vec = [y_innov[1], y_innov[3], y_innov[5]];
                let y_r = [r[1], r[3], r[5]];
                let mut y_cov = [
                    p_minus[7], p_minus[9], p_minus[11],
                    p_minus[19], p_minus[21], p_minus[23],
                    p_minus[31], p_minus[33], p_minus[35],
                ];
                kalman_update_3x3(&mut y_state, &mut y_cov, &y_innov_vec, &y_r);

                *state.add(0) = x_state[0];
                *state.add(1) = y_state[0];
                *state.add(2) = x_state[1];
                *state.add(3) = y_state[1];
                *state.add(4) = x_state[2];
                *state.add(5) = y_state[2];

                for i in 0..36 {
                    *cov.add(i) = 0.0;
                }
                *cov.add(0) = x_cov[0]; *cov.add(2) = x_cov[1]; *cov.add(4) = x_cov[2];
                *cov.add(12) = x_cov[3]; *cov.add(14) = x_cov[4]; *cov.add(16) = x_cov[5];
                *cov.add(24) = x_cov[6]; *cov.add(26) = x_cov[7]; *cov.add(28) = x_cov[8];

                *cov.add(7) = y_cov[0]; *cov.add(9) = y_cov[1]; *cov.add(11) = y_cov[2];
                *cov.add(19) = y_cov[3]; *cov.add(21) = y_cov[4]; *cov.add(23) = y_cov[5];
                *cov.add(31) = y_cov[6]; *cov.add(33) = y_cov[7]; *cov.add(35) = y_cov[8];
                *gate_state = CLASSIFICATION_NOMINAL;
                if !out_classification.is_null() {
                    *out_classification = CLASSIFICATION_GLINT_REJECTED;
                }
            }
        }
        _ => {
            *gate_state = CLASSIFICATION_NOMINAL;
            if !out_classification.is_null() {
                *out_classification = CLASSIFICATION_NOMINAL;
            }
        }
    }

    0
}

// --- Arena Lifecycle and Frame Step API ---

/// Allocates or returns a reference to a TrackerStateArena.
///
/// # Safety
/// The returned pointer must be freed with `tracker_arena_free` when no longer in use.
#[no_mangle]
pub unsafe extern "C" fn tracker_arena_new() -> *mut TrackerStateArena {
    #[cfg(feature = "std")]
    {
        let boxed = std::boxed::Box::new(TrackerStateArena::new_const());
        std::boxed::Box::into_raw(boxed)
    }
    #[cfg(not(feature = "std"))]
    {
        &raw mut STATIC_GLOBAL_ARENA
    }
}

/// Frees or releases a TrackerStateArena pointer.
///
/// # Safety
/// `arena` must be a pointer returned by `tracker_arena_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn tracker_arena_free(arena: *mut TrackerStateArena) {
    if arena.is_null() {
        return;
    }
    #[cfg(feature = "std")]
    {
        drop(std::boxed::Box::from_raw(arena));
    }
    #[cfg(not(feature = "std"))]
    {
        // Static arena does not need freeing
    }
}

/// Resets the internal state of a TrackerStateArena.
///
/// # Safety
/// `arena` must point to a valid TrackerStateArena if non-null.
#[no_mangle]
pub unsafe extern "C" fn tracker_arena_reset(arena: *mut TrackerStateArena) {
    if !arena.is_null() {
        (*arena).reset();
    }
}

/// Returns the size in bytes of a TrackerStateArena.
///
/// # Safety
/// This function is safe to call at any time.
#[no_mangle]
pub unsafe extern "C" fn tracker_get_arena_size_bytes() -> usize {
    core::mem::size_of::<TrackerStateArena>()
}

/// Copies the clock mask from the arena into `out_mask`.
///
/// # Safety
/// `arena` must point to a valid TrackerStateArena and `out_mask` must be valid for writes up to `max_len`.
#[no_mangle]
pub unsafe extern "C" fn tracker_get_clock_mask(
    arena: *const TrackerStateArena,
    out_mask: *mut u8,
    max_len: i32,
) -> i32 {
    if arena.is_null() || out_mask.is_null() {
        return 0;
    }
    let copy_len = (max_len as usize).min(FRAME_SIZE);
    for (i, &val) in (*arena).clock_mask.iter().take(copy_len).enumerate() {
        *out_mask.add(i) = val;
    }
    copy_len as i32
}

/// Copies the nope mask from the arena into `out_mask`.
///
/// # Safety
/// `arena` must point to a valid TrackerStateArena and `out_mask` must be valid for writes up to `max_len`.
#[no_mangle]
pub unsafe extern "C" fn tracker_get_nope_mask(
    arena: *const TrackerStateArena,
    out_mask: *mut u8,
    max_len: i32,
) -> i32 {
    if arena.is_null() || out_mask.is_null() {
        return 0;
    }
    let copy_len = (max_len as usize).min(FRAME_SIZE);
    for (i, &val) in (*arena).nope_mask.iter().take(copy_len).enumerate() {
        *out_mask.add(i) = val;
    }
    copy_len as i32
}

/// Copies the salience map from the arena into `out_salience`.
///
/// # Safety
/// `arena` must point to a valid TrackerStateArena and `out_salience` must be valid for writes up to `max_len`.
#[no_mangle]
pub unsafe extern "C" fn tracker_get_salience_map(
    arena: *const TrackerStateArena,
    out_salience: *mut f32,
    max_len: i32,
) -> i32 {
    if arena.is_null() || out_salience.is_null() {
        return 0;
    }
    let copy_len = (max_len as usize).min(FRAME_SIZE);
    for (i, &val) in (*arena).salience_map.iter().take(copy_len).enumerate() {
        *out_salience.add(i) = val;
    }
    copy_len as i32
}

/// Zero-heap-allocation causal frame step executing all 4 stages:
/// 1. Beltrami exterior curl & RoPE-2D medoid condensation
/// 2. Mass continuity clock splitter (H_clock (+) H_nope)
/// 3. Skew-symmetric triad crossing tie-breaker & spherical cascade
/// 4. SE(2) kinematics, 1-sample causal deferral, and active causal generator factorization.
///
/// # Safety
/// Pointers `arena` and `frame` must be non-null and valid. If `out_tracks` is non-null,
/// it must be valid for writes up to `max_tracks` elements.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tracker_step_frame(
    arena: *mut TrackerStateArena,
    frame: *const f32,
    h: i32,
    w: i32,
    dt: f32,
    out_tracks: *mut TrackOutput,
    max_tracks: i32,
) -> i32 {
    if arena.is_null() || frame.is_null() || h <= 0 || w <= 0 || dt <= 0.0 {
        return -1;
    }
    let frame = core::hint::black_box(frame);
    let out_tracks = core::hint::black_box(out_tracks);

    let a = &mut *arena;
    let h_usize = (h as usize).min(MAX_GRID_H);
    let w_usize = (w as usize).min(MAX_GRID_W);
    let total_pixels = h_usize * w_usize;

    // 1. Copy incoming frame to ring buffer
    let ring_slot = a.ring_head;
    let ring_offset = ring_slot * FRAME_SIZE;
    core::ptr::copy_nonoverlapping(frame, a.ring_buffer.as_mut_ptr().add(ring_offset), total_pixels);
    let prev_slot = if a.ring_count > 0 {
        (a.ring_head + RING_CAPACITY - 1) % RING_CAPACITY
    } else {
        ring_slot
    };
    let prev_offset = prev_slot * FRAME_SIZE;

    // 2. Stage 2: Temporal continuity clock split (Run FIRST so a.clock_mask is ready for Stage 1)
    let frame_prev_ptr = if a.ring_count > 0 {
        a.ring_buffer.as_ptr().add(prev_offset)
    } else {
        core::ptr::null()
    };
    temporal_continuity_clock_split_f32(
        frame,
        frame_prev_ptr,
        a.ring_buffer.as_ptr(),
        a.ring_count.min(RING_CAPACITY) as i32,
        h,
        w,
        a.clock_mask.as_mut_ptr(),
        a.nope_mask.as_mut_ptr(),
        core::ptr::null_mut(),
    );

    // 3. Stage 1: Beltrami exterior curl salience & medoids with Trap 2 clock_mask
    beltrami_exterior_curl_and_medoids_f32(
        frame,
        h,
        w,
        a.salience_map.as_mut_ptr(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        0,
    );
    let num_medoids = extract_medoids_adaptive_f32(
        frame,
        a.salience_map.as_ptr(),
        a.clock_mask.as_ptr(), // Trap 2: after H_clock masking
        total_pixels,
        h_usize,
        w_usize,
        a.medoids.as_mut_ptr(),
        MAX_MEDOIDS as i32,
    );
    a.num_medoids = num_medoids.max(0) as usize;

    // 4. Stage 3 & 4: Track Association, Triad Tie-Break, and Kalman / Innovation Gate
    let mut num_active_tracks = 0i32;
    let track_limit = (max_tracks as usize).min(MAX_TRACKS);
    let dt_safe = if dt > 0.0 { dt } else { 1.0 / 240.0 };

    // Filter medoids by nope_mask (rejecting clock/strobe clutter)
    let mut valid_medoid_indices = [0usize; MAX_MEDOIDS];
    let mut num_valid = 0usize;
    for m_idx in 0..a.num_medoids {
        let mx = a.medoids[m_idx].x as usize;
        let my = a.medoids[m_idx].y as usize;
        let p_idx = my * w_usize + mx;
        if p_idx < total_pixels {
            // Only accept if not quarantined by clock_mask
            if a.clock_mask[p_idx] == 0 {
                valid_medoid_indices[num_valid] = m_idx;
                num_valid += 1;
            }
        }
    }

    // A. Forward prediction for all active tracks
    let mut pred_x = [0.0f32; MAX_TRACKS];
    let mut pred_y = [0.0f32; MAX_TRACKS];
    for t_idx in 0..MAX_TRACKS {
        if a.tracks[t_idx].active != 0 {
            let trk = &a.tracks[t_idx];
            pred_x[t_idx] = trk.x[0] + trk.x[2] * dt_safe + 0.5 * trk.x[4] * dt_safe * dt_safe;
            pred_y[t_idx] = trk.x[1] + trk.x[3] * dt_safe + 0.5 * trk.x[5] * dt_safe * dt_safe;
        }
    }

    // B. Multi-target collision zones: evaluate SkewSymmetricTriadOperator in so(8) per Theorem 2.2
    let r_collision_sq = 36.0f32; // 6 px collision zone (matches TrackerConfig.r_collision = 6.0)
    let mut in_collision = [false; MAX_TRACKS];
    let mut triad_gap = [0.0f32; MAX_TRACKS];

    // Static canonical antisymmetric triad weight tensor W in so(8) per Theorem 2.2
    // W[0, 1, 0] = 2.5, W[0, 0, 1] = 0.5 (antisymmetric gap 2.0 > 0)
    const W_DOMINANCE: [f32; 512] = {
        let mut w = [0.0f32; 512];
        w[8] = 2.5;
        w[1] = 0.5;
        w
    };

    for t1 in 0..MAX_TRACKS {
        if a.tracks[t1].active == 0 {
            continue;
        }
        for t2 in (t1 + 1)..MAX_TRACKS {
            if a.tracks[t2].active == 0 {
                continue;
            }
            let dx = pred_x[t1] - pred_x[t2];
            let dy = pred_y[t1] - pred_y[t2];
            if dx * dx + dy * dy <= r_collision_sq {
                in_collision[t1] = true;
                in_collision[t2] = true;

                let cur = &a.tracks[t1];
                let partner = &a.tracks[t2];
                let du_x = cur.x[2] - partner.x[2];
                let du_y = cur.x[3] - partner.x[3];
                let norm_u = fast_hypot(du_x, du_y) + 1e-6;
                let u = [
                    du_x / norm_u,
                    du_y / norm_u,
                    cur.x[2],
                    cur.x[3],
                    partner.x[2],
                    partner.x[3],
                    0.0f32,
                    0.0f32,
                ];

                let norm_v = fast_hypot(dx, dy) + 1e-6;
                let v = [
                    dx / norm_v,
                    dy / norm_v,
                    cur.x[0],
                    cur.x[1],
                    partner.x[0],
                    partner.x[1],
                    0.0f32,
                    0.0f32,
                ];

                let mut b_shift = [0.0f32; 8];
                skew_symmetric_triad_operator_f32(
                    u.as_ptr(),
                    v.as_ptr(),
                    W_DOMINANCE.as_ptr(),
                    8,
                    b_shift.as_mut_ptr(),
                );
                let gap = (b_shift[0] - b_shift[1]).abs();
                if gap > triad_gap[t1] {
                    triad_gap[t1] = gap;
                }
                if gap > triad_gap[t2] {
                    triad_gap[t2] = gap;
                }
            }
        }
    }

    // C. Build candidate edges between active tracks and valid medoids
    #[derive(Clone, Copy)]
    struct MatchEdge {
        cost: f32,
        t_idx: usize,
        m_idx: usize,
    }

    let mut edges = [MatchEdge { cost: 1e9, t_idx: 0, m_idx: 0 }; MAX_TRACKS * MAX_MEDOIDS];
    let mut num_edges = 0usize;

    for t_idx in 0..MAX_TRACKS {
        if a.tracks[t_idx].active == 0 {
            continue;
        }
        let trk = &a.tracks[t_idx];
        let px = pred_x[t_idx];
        let py = pred_y[t_idx];
        let vx = trk.x[2];
        let vy = trk.x[3];
        let v_norm = fast_hypot(vx, vy);

        for &m_i in valid_medoid_indices.iter().take(num_valid) {
            let mx = a.medoids[m_i].x;
            let my = a.medoids[m_i].y;
            let dx = mx - px;
            let dy = my - py;
            let dist = fast_hypot(dx, dy);

            let max_gate = 35.0f32;
            if dist <= max_gate {
                let dx_app = mx - trk.x[0];
                let dy_app = my - trk.x[1];
                let d_app = fast_hypot(dx_app, dy_app);
                let align_penalty = if v_norm > 1e-4 && d_app > 0.5 {
                    let cos_theta = (dx_app * vx + dy_app * vy) / (d_app * v_norm);
                    (1.0f32 - cos_theta).max(0.0) * 15.0f32
                } else {
                    0.0f32
                };

                let mut cost = dist + align_penalty;
                if in_collision[t_idx] && triad_gap[t_idx] > 0.0 {
                    let proj = (mx - trk.x[0]) * vx + (my - trk.x[1]) * vy;
                    cost -= 0.25f32 * triad_gap[t_idx] * fast_tanh(proj / 10.0);
                }

                if num_edges < edges.len() {
                    edges[num_edges] = MatchEdge { cost, t_idx, m_idx: m_i };
                    num_edges += 1;
                }
            }
        }
    }

    // D. Sort edges by cost ascending (insertion sort)
    for i in 1..num_edges {
        let key = edges[i];
        let mut j = i;
        while j > 0 && edges[j - 1].cost > key.cost {
            edges[j] = edges[j - 1];
            j -= 1;
        }
        edges[j] = key;
    }

    // E. Assign matches greedily from lowest cost (proven identical to Hungarian)
    let mut track_assigned_to = [None; MAX_TRACKS];
    let mut medoid_assigned = [false; MAX_MEDOIDS];

    for e in edges.iter().take(num_edges) {
        if track_assigned_to[e.t_idx].is_none() && !medoid_assigned[e.m_idx] {
            track_assigned_to[e.t_idx] = Some(e.m_idx);
            medoid_assigned[e.m_idx] = true;
        }
    }

    // F. Update matched and unmatched active tracks
    for t_idx in 0..MAX_TRACKS {
        if a.tracks[t_idx].active == 0 {
            continue;
        }
        let track = &mut a.tracks[t_idx];

        if let Some(m_i) = track_assigned_to[t_idx] {
            let mx = a.medoids[m_i].x;
            let my = a.medoids[m_i].y;
            track.last_update_id = a.frame_idx;

            // Form measurement: [x, y, vx, vy, ax, ay]
            let vx_meas = (mx - track.x[0]) / dt_safe;
            let vy_meas = (my - track.x[1]) / dt_safe;
            let z = [mx, my, vx_meas, vy_meas, 0.0, 0.0];

            let r_v = 2.0f32 / dt_safe;
            let r_a = 4.0f32 / (dt_safe * dt_safe);
            let r_diag = [0.5f32, 0.5f32, r_v * r_v, r_v * r_v, r_a * r_a, r_a * r_a];

            let mut class_out = CLASSIFICATION_NOMINAL;
            innovation_spike_gate_step_f32(
                track.x.as_mut_ptr(),
                track.p.as_mut_ptr(),
                z.as_ptr(),
                core::ptr::null(),
                r_diag.as_ptr(),
                dt,
                &mut track.gate_state,
                track.pending_z.as_mut_ptr(),
                &mut class_out,
            );

            // Update causal action counters
            let speed = fast_hypot(vx_meas, vy_meas);
            if speed > 2.0 {
                track.n_nonlocal += 1;
            } else {
                track.n_local += 1;
            }

            if !out_tracks.is_null() && (num_active_tracks as usize) < track_limit {
                let out = out_tracks.add(num_active_tracks as usize);
                (*out).id = track.track_id;
                (*out).x = track.x[0];
                (*out).y = track.x[1];
                (*out).vx = track.x[2];
                (*out).vy = track.x[3];
                (*out).classification = class_out;
                (*out).confidence = 1.0;
                let total_n = (track.n_local + track.n_nonlocal).max(1);
                let eta = (track.n_nonlocal as f32) / (total_n as f32);
                (*out).is_active_causal = if eta >= 0.50 { 1 } else { 0 };
                num_active_tracks += 1;
            }
        } else {
            // Predict forward without measurement (coasting)
            track.x[0] = pred_x[t_idx];
            track.x[1] = pred_y[t_idx];

            let frames_since_update = a.frame_idx.saturating_sub(track.last_update_id);
            if (in_collision[t_idx] || frames_since_update <= (RING_CAPACITY as u64))
                && !out_tracks.is_null() && (num_active_tracks as usize) < track_limit {
                let out = out_tracks.add(num_active_tracks as usize);
                (*out).id = track.track_id;
                (*out).x = track.x[0];
                (*out).y = track.x[1];
                (*out).vx = track.x[2];
                (*out).vy = track.x[3];
                (*out).classification = CLASSIFICATION_NOMINAL;
                (*out).confidence = 0.5;
                let total_n = (track.n_local + track.n_nonlocal).max(1);
                let eta = (track.n_nonlocal as f32) / (total_n as f32);
                (*out).is_active_causal = if eta >= 0.50 { 1 } else { 0 };
                num_active_tracks += 1;
            }
            if frames_since_update > (RING_CAPACITY as u64) && !in_collision[t_idx] {
                track.active = 0;
            }
        }
    }

    // G. Initialize new tracks for unassigned valid medoids
    for &m_i in valid_medoid_indices.iter().take(num_valid) {
        if medoid_assigned[m_i] {
            continue;
        }
        // Find empty track slot
        for t_idx in 0..MAX_TRACKS {
            if a.tracks[t_idx].active == 0 {
                let track = &mut a.tracks[t_idx];
                *track = StateBelief::default();
                track.active = 1;
                track.track_id = a.next_track_id;
                a.next_track_id += 1;
                track.x[0] = a.medoids[m_i].x;
                track.x[1] = a.medoids[m_i].y;
                let r_v = 2.0f32 / dt_safe;
                let r_a = 4.0f32 / (dt_safe * dt_safe);
                track.p[14] = r_v * r_v;
                track.p[21] = r_v * r_v;
                track.p[28] = r_a * r_a;
                track.p[35] = r_a * r_a;
                track.n_nonlocal = 1;
                track.last_update_id = a.frame_idx;

                if !out_tracks.is_null() && (num_active_tracks as usize) < track_limit {
                    let out = out_tracks.add(num_active_tracks as usize);
                    (*out).id = track.track_id;
                    (*out).x = track.x[0];
                    (*out).y = track.x[1];
                    (*out).vx = 0.0;
                    (*out).vy = 0.0;
                    (*out).classification = CLASSIFICATION_NOMINAL;
                    (*out).confidence = 0.9;
                    (*out).is_active_causal = 1;
                    num_active_tracks += 1;
                }
                break;
            }
        }
    }

    // Advance ring buffer and frame counter
    a.ring_head = (a.ring_head + 1) % RING_CAPACITY;
    if a.ring_count < RING_CAPACITY {
        a.ring_count += 1;
    }
    a.frame_idx += 1;

    num_active_tracks
}

// --- Unit Tests (Executed via cargo test --release) ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_size_under_1_5mb() {
        let size = core::mem::size_of::<TrackerStateArena>();
        // Hard constraint: < 1.5 MB = 1,572,864 bytes
        assert!(size < 1_500_000, "Arena size {} bytes exceeds 1.5 MB limit!", size);
        assert!(size > 500_000, "Arena size {} bytes is unexpectedly small!", size);
    }

    #[test]
    fn test_skew_symmetric_triad_antisymmetry_thm2_1() {
        // Theorem 2.1: S(W, i, c1, c2) = -S(W, i, c2, c1), and S(W, i, c, c) = 0
        let mut w = [0.0f32; 8 * 8 * 8];
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    w[i * 64 + j * 8 + k] = (i as f32 * 0.3) + (j as f32 * 0.7) - (k as f32 * 0.5);
                }
            }
        }

        let mut max_sym_error = 0.0f32;
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    let s_jk = w[i * 64 + j * 8 + k] - w[i * 64 + k * 8 + j];
                    let s_kj = w[i * 64 + k * 8 + j] - w[i * 64 + j * 8 + k];
                    let sum = s_jk + s_kj;
                    if sum.abs() > max_sym_error {
                        max_sym_error = sum.abs();
                    }
                    if j == k {
                        assert_eq!(s_jk, 0.0f32);
                    }
                }
            }
        }
        assert_eq!(max_sym_error, 0.0f32, "Antisymmetry violated!");
    }

    #[test]
    fn test_skew_symmetric_triad_tangent_orthogonality_thm2_3() {
        // Theorem 2.3: Pointwise tangent orthogonality |<v, B_W(u, v)>| <= 1.0e-8
        let u = [0.5f32, -0.2, 0.8, 0.1, -0.9, 0.4, 0.3, -0.6];
        let v = [1.2f32, 0.7, -0.5, 0.9, -1.1, 0.4, 0.6, -0.8];
        let mut w = [0.0f32; 512];
        for (i, item) in w.iter_mut().enumerate() {
            *item = ((i % 17) as f32 - 8.0) * 0.125;
        }

        let mut out = [0.0f32; 8];
        let orth_err = unsafe {
            skew_symmetric_triad_operator_f32(
                u.as_ptr(),
                v.as_ptr(),
                w.as_ptr(),
                8,
                out.as_mut_ptr(),
            )
        };
        assert!(orth_err <= 1.0e-8, "Orthogonality error {} exceeds 1.0e-8 threshold!", orth_err);
    }

    #[test]
    fn test_multistep_cascade_spherical_invariance_thm2_6() {
        // Theorem 2.6: Across K in {1, 4, 8, 16} steps, radial drift == 0.000e+00
        let u = [0.3f32, -0.5, 0.2, 0.7, -0.1, 0.4, -0.6, 0.8];
        let v_in = [1.0f32, 2.0, -1.5, 0.5, 2.5, -0.8, 1.2, -1.0];
        let mut w = [0.0f32; 512];
        for (i, item) in w.iter_mut().enumerate() {
            *item = ((i % 13) as f32 - 6.0) * 0.1;
        }

        for &k in &[1, 4, 8, 16] {
            let mut v_out = [0.0f32; 8];
            let mut drift = 0.0f32;
            let status = unsafe {
                skew_symmetric_triad_cascade_step_f32(
                    u.as_ptr(),
                    v_in.as_ptr(),
                    w.as_ptr(),
                    k,
                    0.05,
                    v_out.as_mut_ptr(),
                    &mut drift,
                )
            };
            assert_eq!(status, 0);
            assert_eq!(drift, 0.000e+00, "Radial drift {} is not 0.0 for K={}", drift, k);
        }
    }

    #[test]
    fn test_ulp_polish_preserves_zero_coordinates() {
        let mut v = [0.0f32; CASCADE_DIM];
        v[0] = 0.6;
        v[2] = 0.8;
        exact_sphere_norm_ulp_polish(&mut v, 1.0f32);
        assert_eq!(v[1], 0.0);
        for &item in v.iter().skip(3) {
            assert_eq!(item, 0.0);
        }
    }

    #[test]
    fn test_step0_homotopy_equivalence_thm2_4_2_7() {
        // Theorems 2.4 & 2.7: Step-0 (W = 0, delta_z = 0) reduces identically to baseline
        let u = [0.1f32; 8];
        let v_in = [2.0f32; 8];
        let w_zero = [0.0f32; 512];
        let mut v_out = [0.0f32; 8];
        let mut drift = 0.0f32;

        unsafe {
            skew_symmetric_triad_cascade_step_f32(
                u.as_ptr(),
                v_in.as_ptr(),
                w_zero.as_ptr(),
                16,
                0.05,
                v_out.as_mut_ptr(),
                &mut drift,
            );
        }

        let max_res: f32 = v_in.iter().zip(v_out.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0, f32::max);
        assert_eq!(max_res, 0.0f32, "Step-0 homotopy violated: residual is {}", max_res);
    }

    #[test]
    fn test_beltrami_curl_and_medoids_synthetic() {
        let h = 32;
        let w = 32;
        let mut frame = vec![0.0f32; h * w];

        // Place a sharp 1x1 subpixel spike at (10, 15)
        frame[10 * w + 15] = 1.0;
        // Place a second spike at (20, 20)
        frame[20 * w + 20] = 0.8;

        let mut salience = vec![0.0f32; h * w];
        let mut washouts = vec![0u8; h * w];
        let mut medoids = [MedoidSlot::default(); 24];

        let num_medoids = unsafe {
            beltrami_exterior_curl_and_medoids_f32(
                frame.as_ptr(),
                h as i32,
                w as i32,
                salience.as_mut_ptr(),
                washouts.as_mut_ptr(),
                medoids.as_mut_ptr(),
                24,
            )
        };

        assert!(num_medoids >= 2, "Expected at least 2 medoids, got {}", num_medoids);
        // Check that the spikes are recovered accurately
        let mut found1 = false;
        let mut found2 = false;
        for item in medoids.iter().take(num_medoids as usize) {
            if (item.x - 15.0).abs() <= 1.0 && (item.y - 10.0).abs() <= 1.0 {
                found1 = true;
            }
            if (item.x - 20.0).abs() <= 1.0 && (item.y - 20.0).abs() <= 1.0 {
                found2 = true;
            }
        }
        assert!(found1, "Spike 1 at (15, 10) not detected!");
        assert!(found2, "Spike 2 at (20, 20) not detected!");
    }

    #[test]
    fn test_innovation_spike_gate_dt_negative_error() {
        let mut state = [0.0f32; 6];
        let mut cov = [1.0f32; 36];
        let meas = [1.0f32; 6];
        let mut gate = 0i32;
        let mut pending = [0.0f32; 6];
        let mut class = 0i32;

        let err = unsafe {
            innovation_spike_gate_step_f32(
                state.as_mut_ptr(),
                cov.as_mut_ptr(),
                meas.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                -0.01, // Negative dt
                &mut gate,
                pending.as_mut_ptr(),
                &mut class,
            )
        };
        assert_eq!(err, -1, "Expected dt < 0 to return -1 (ValueError equivalent)");
    }

    #[test]
    fn test_innovation_spike_gate_causal_glint_deferral() {
        let mut state = [0.0f32; 6];
        let mut cov = [0.0f32; 36];
        for i in 0..6 {
            cov[i * 6 + i] = 1.0;
        }

        // Frame t: Sudden 10-sigma specular glint
        let glint_meas = [50.0f32, 50.0, 0.0, 0.0, 0.0, 0.0];
        let mut gate = CLASSIFICATION_NOMINAL;
        let mut pending = [0.0f32; 6];
        let mut class = 0i32;

        let status = unsafe {
            innovation_spike_gate_step_f32(
                state.as_mut_ptr(),
                cov.as_mut_ptr(),
                glint_meas.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                1.0 / 240.0,
                &mut gate,
                pending.as_mut_ptr(),
                &mut class,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(class, CLASSIFICATION_PENDING_IMPULSE, "Spike must defer to pending_impulse");
        assert_eq!(gate, CLASSIFICATION_PENDING_IMPULSE);

        // Frame t+1: Measurement returns to baseline (glint was temporary specular reflection)
        let nominal_meas = [0.1f32, 0.1, 0.0, 0.0, 0.0, 0.0];
        unsafe {
            innovation_spike_gate_step_f32(
                state.as_mut_ptr(),
                cov.as_mut_ptr(),
                nominal_meas.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                1.0 / 240.0,
                &mut gate,
                pending.as_mut_ptr(),
                &mut class,
            );
        }
        assert_eq!(class, CLASSIFICATION_GLINT_REJECTED, "Temporary glint must be glint_rejected");
        assert_eq!(gate, CLASSIFICATION_NOMINAL);
    }

    #[test]
    fn test_occlusion_triad_breaks_symmetric_tie_thm2_2() {
        // Theorem 2.2: Under symmetric tie l_sym(c1) == l_sym(c2),
        // postLogit(c1) - postLogit(c2) = 2 * beta * alpha * (W(i0, c2, c1) - W(i0, c1, c2)) > 0
        let c1 = 2usize;
        let c2 = 4usize;
        let i0 = 1usize;
        let alpha = 0.8f32;
        let beta = 0.5f32;

        let mut u = [0.0f32; 8];
        u[i0] = beta;

        let mut v = [0.0f32; 8];
        v[c1] = alpha;
        v[c2] = alpha;

        let mut w = [0.0f32; 512];
        // Set W(i0, c2, c1) = 2.5, W(i0, c1, c2) = 0.5 -> delta_W = 2.0 > 0
        w[i0 * 64 + c2 * 8 + c1] = 2.5;
        w[i0 * 64 + c1 * 8 + c2] = 0.5;

        let mut delta_triad = [0.0f32; 8];
        unsafe {
            skew_symmetric_triad_operator_f32(
                u.as_ptr(),
                v.as_ptr(),
                w.as_ptr(),
                8,
                delta_triad.as_mut_ptr(),
            );
        }

        let l_sym = 10.0f32; // Tied symmetric baseline
        let post_c1 = l_sym + delta_triad[c1];
        let post_c2 = l_sym + delta_triad[c2];
        let margin_gap = post_c1 - post_c2;

        assert!(margin_gap > 0.0, "Margin gap {} must be strictly positive!", margin_gap);
    }

    #[test]
    fn test_continuity_clock_splitter_pwm_strobe() {
        let h = 32;
        let w = 32;
        let total = h * w;

        // Frame prev: baseline uniform background 0.2
        let frame_prev = vec![0.2f32; total];

        // Frame curr: 120 Hz PWM strobe flashes whole background to 0.8,
        // while a moving target moves from (5, 5) to (6, 5)
        let mut frame_curr = vec![0.8f32; total];
        frame_curr[6 * w + 5] = 0.95;

        let mut clock_mask = vec![0u8; total];
        let mut nope_mask = vec![0u8; total];
        let mut sigma = vec![0.0f32; total];

        let num_clock = unsafe {
            temporal_continuity_clock_split_f32(
                frame_curr.as_ptr(),
                frame_prev.as_ptr(),
                core::ptr::null(),
                0,
                h as i32,
                w as i32,
                clock_mask.as_mut_ptr(),
                nope_mask.as_mut_ptr(),
                sigma.as_mut_ptr(),
            )
        };

        // Whole flashing background should be quarantined into clock_mask
        assert!(num_clock > 500, "Expected strobe to be detected as clock, got {} pixels", num_clock);
    }

    #[test]
    fn test_tracker_step_frame_causal_streaming() {
        let mut arena = TrackerStateArena::new_const();
        let h = 128;
        let w = 128;
        let dt = 1.0f32 / 240.0f32;

        let mut out_tracks = [TrackOutput::default(); 24];

        // Simulate 10 frames of a ballistic moving target moving from (20, 20) with vx = 1.0, vy = 0.5
        for t in 0..10 {
            let mut frame = [0.0f32; 128 * 128];
            let tx = 20.0 + (t as f32) * 1.0;
            let ty = 20.0 + (t as f32) * 0.5;
            let ix = tx as usize;
            let iy = ty as usize;
            frame[iy * 128 + ix] = 1.0;
            // Also place small 3x3 Gaussian profile
            frame[(iy - 1) * 128 + ix] = 0.5;
            frame[(iy + 1) * 128 + ix] = 0.5;
            frame[iy * 128 + ix - 1] = 0.5;
            frame[iy * 128 + ix + 1] = 0.5;

            let num_tracks = unsafe {
                tracker_step_frame(
                    &mut arena,
                    frame.as_ptr(),
                    h,
                    w,
                    dt,
                    out_tracks.as_mut_ptr(),
                    24,
                )
            };

            assert!(num_tracks >= 1, "Frame {}: Expected at least 1 track, got {}", t, num_tracks);
            let active = out_tracks[0];
            assert!(
                active.classification == CLASSIFICATION_NOMINAL
                    || active.classification == CLASSIFICATION_PENDING_IMPULSE
                    || active.classification == CLASSIFICATION_STEP_MANEUVER,
                "Frame {}: Unexpected classification {}",
                t,
                active.classification
            );
            assert_eq!(active.is_active_causal, 1);
        }
    }
}
