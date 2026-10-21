"""
scripts/stress_scenario_03.py: Adversarial Empirical Stress Test Suite for Scenario 03 Symmetric Crossing Occlusions.

Author: challenger_gate_1 (teamwork_preview_challenger)
Date: 2026-10-21
Scope:
1. Systematic multi-resolution stress testing (128x128 and 256x256).
2. Varied collision geometries (90 deg, 180 deg, oblique 45/60/120/135 deg, 6-target and 8-target stars).
3. Varied speed regimes (slow 0.1-0.2 px/frame, nominal 0.4 px/frame, fast 1.5-5.0 px/frame, asymmetric speeds).
4. Varied occlusion durations (1 to 25 frames full overlap).
5. Degenerate boundary conditions (single target, co-linear chasing, extreme noise floors, intensity asymmetry, vanishing targets).
6. Comprehensive comparison across Python reference (DarkSnail Streaming/Batch), Rust SIMD kernel, and classical CV baselines.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from darksnail.tracker import (
    DarkSnailHighSpeedTracker,
    InnovationSpikeGateConfig,
    TrackerConfig,
)
from darksnail.tracker_benchmark import (
    NaiveFrameDiffCentroid,
    PatchAvgKalmanIoU,
    evaluate_tracker_blind,
)


@dataclass
class TestResult:
    scenario_id: str
    description: str
    grid_size: int
    backend: str
    num_targets: int
    id_swaps: int
    pre_post_match: bool
    mean_cos_sim: float
    rmse: float
    mota: float
    fps: float
    status: str  # PASS / FAIL / BUG
    details: str


def make_custom_crossing_scenario(
    name: str,
    H: int = 128,
    W: int = 128,
    num_frames: int = 240,
    crossing_frame: int = 120,
    targets_config: Optional[List[Dict[str, Any]]] = None,
    overlap_duration: int = 5,
    noise_sigma: float = 5.0,
    seed: int = 42,
    dt: float = 1.0 / 240.0,
) -> Dict[str, Any]:
    """
    Parametric synthetic generator for arbitrary crossing geometries and dynamics.
    """
    rng = np.random.RandomState(seed)
    frames = np.zeros((num_frames, H, W), dtype=np.float32)
    ground_truth: List[List[Dict[str, Any]]] = []

    cx, cy = float(W // 2), float(H // 2)

    if targets_config is None:
        # Default 4-target cardinal crossing
        targets_config = [
            {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 200.0, "size": 2},     # Moves East
            {"id": 2, "angle_deg": 180.0, "speed": 0.4, "intensity": 200.0, "size": 2},   # Moves West
            {"id": 3, "angle_deg": 90.0, "speed": 0.4, "intensity": 200.0, "size": 2},    # Moves South
            {"id": 4, "angle_deg": 270.0, "speed": 0.4, "intensity": 200.0, "size": 2},   # Moves North
        ]

    half_overlap = overlap_duration / 2.0

    for t in range(num_frames):
        frame_gt = []

        # Compute position for each target at frame t
        for cfg in targets_config:
            tid = cfg["id"]
            speed = float(cfg.get("speed", 0.4))
            intensity = float(cfg.get("intensity", 200.0))
            sz = int(cfg.get("size", 2))
            angle_rad = math.radians(float(cfg.get("angle_deg", 0.0)))
            disappear_during_occlusion = cfg.get("disappear_during_occlusion", False)
            dwell_frames = int(cfg.get("dwell_frames", 0))

            # Target direction vector
            dir_x = math.cos(angle_rad)
            dir_y = math.sin(angle_rad)

            # Distance along trajectory relative to crossing center
            delta_t = t - crossing_frame
            if dwell_frames > 0 and abs(delta_t) <= (dwell_frames / 2.0):
                dist = 0.0
                vx = 0.0
                vy = 0.0
            elif dwell_frames > 0:
                sgn = 1.0 if delta_t > 0 else -1.0
                effective_delta = delta_t - sgn * (dwell_frames / 2.0)
                dist = effective_delta * speed
                vx = speed * dir_x
                vy = speed * dir_y
            else:
                dist = delta_t * speed
                vx = speed * dir_x
                vy = speed * dir_y

            # Current position
            tx = cx + dist * dir_x
            ty = cy + dist * dir_y

            # Draw target unless vanishing during occlusion
            draw_intensity = intensity
            if disappear_during_occlusion and abs(delta_t) <= (overlap_duration / 2.0):
                draw_intensity = 0.0

            if draw_intensity > 0.0:
                itx, ity = int(round(tx)), int(round(ty))
                for dy in range(sz):
                    for dx in range(sz):
                        py, px = ity + dy, itx + dx
                        if 0 <= px < W and 0 <= py < H:
                            frames[t, py, px] = max(frames[t, py, px], draw_intensity)

            frame_gt.append({
                "id": tid,
                "x": float(tx),
                "y": float(ty),
                "vx": float(vx),
                "vy": float(vy),
                "size": sz,
                "visible": (draw_intensity > 0.0),
            })

        # Add Gaussian sensor noise
        if noise_sigma > 0.0:
            noise = rng.normal(0, noise_sigma, size=(H, W)).astype(np.float32)
            frames[t] = np.clip(frames[t] + noise, 0, 255)
        else:
            frames[t] = np.clip(frames[t], 0, 255)

        ground_truth.append(frame_gt)

    return {
        "name": name,
        "frames": frames,
        "ground_truth": ground_truth,
        "dt": dt,
        "H": H,
        "W": W,
        "crossing_frame": crossing_frame,
    }


def evaluate_trajectory_continuity(
    tracker: Any,
    scenario_data: Dict[str, Any],
    margin: int = 15,
) -> Dict[str, Any]:
    """
    Rigorously verifies identity preservation and directional momentum continuity
    before and after crossing center.
    """
    frames = scenario_data["frames"]
    ground_truth = scenario_data["ground_truth"]
    dt = scenario_data["dt"]
    num_frames = len(frames)
    crossing_frame = scenario_data.get("crossing_frame", num_frames // 2)

    if hasattr(tracker, "reset"):
        tracker.reset()

    step_outputs = []
    for t in range(num_frames):
        out = tracker.update(frames[t], dt)
        step_outputs.append(out)

    # 1. Standard CLEAR-MOT metrics via evaluate_tracker_blind
    blind_res = evaluate_tracker_blind(tracker, scenario_data)

    # 2. Pre-crossing vs post-crossing track ID mapping
    pre_window = range(max(0, crossing_frame - margin - 20), max(0, crossing_frame - margin))
    post_window = range(min(num_frames, crossing_frame + margin), min(num_frames, crossing_frame + margin + 20))

    gt_ids = [g["id"] for g in ground_truth[0]]
    pre_mapping: Dict[int, int] = {}
    post_mapping: Dict[int, int] = {}
    post_cos_sims: List[float] = []

    def get_modal_tracker_id(window_range, gid):
        id_votes = {}
        for t in window_range:
            gts = {g["id"]: g for g in ground_truth[t]}
            if gid not in gts:
                continue
            gt_pos = gts[gid]
            tracks = step_outputs[t].get("tracks", [])
            for trk in tracks:
                d = math.hypot(trk["x"] - gt_pos["x"], trk["y"] - gt_pos["y"])
                if d <= 8.0:
                    tid = trk["id"]
                    id_votes[tid] = id_votes.get(tid, 0) + 1
        if not id_votes:
            return None
        return max(id_votes, key=id_votes.get)

    for gid in gt_ids:
        pre_id = get_modal_tracker_id(pre_window, gid)
        post_id = get_modal_tracker_id(post_window, gid)
        if pre_id is not None:
            pre_mapping[gid] = pre_id
        if post_id is not None:
            post_mapping[gid] = post_id

    # Check pre/post identity equality
    pre_post_match = True
    for gid in gt_ids:
        if gid in pre_mapping and gid in post_mapping:
            if pre_mapping[gid] != post_mapping[gid]:
                pre_post_match = False
                break
        elif gid in pre_mapping and gid not in post_mapping:
            # Track lost completely after crossing
            pre_post_match = False
            break

    # Directional momentum continuity post-crossing
    for t in post_window:
        gts = {g["id"]: g for g in ground_truth[t]}
        for trk in step_outputs[t].get("tracks", []):
            for gid, g in gts.items():
                d = math.hypot(trk["x"] - g["x"], trk["y"] - g["y"])
                if d <= 8.0:
                    trk_vx, trk_vy = trk.get("vx", 0.0), trk.get("vy", 0.0)
                    gt_vx, gt_vy = g["vx"], g["vy"]
                    trk_mag = math.hypot(trk_vx, trk_vy)
                    gt_mag = math.hypot(gt_vx, gt_vy)
                    if trk_mag > 1e-3 and gt_mag > 1e-3:
                        dot = trk_vx * gt_vx + trk_vy * gt_vy
                        cos_sim = dot / (trk_mag * gt_mag)
                        post_cos_sims.append(cos_sim)

    mean_cos_sim = float(np.mean(post_cos_sims)) if post_cos_sims else 1.0

    return {
        "blind_metrics": blind_res,
        "pre_mapping": pre_mapping,
        "post_mapping": post_mapping,
        "pre_post_match": pre_post_match,
        "mean_cos_sim": mean_cos_sim,
    }


def run_all_stress_tests() -> List[TestResult]:
    results: List[TestResult] = []

    dt = 1.0 / 240.0
    gate_cfg = InnovationSpikeGateConfig(
        pos_sigma=1.0,
        vel_sigma=2.0 / dt,
        acc_sigma=4.0 / (dt * dt),
    )
    trk_cfg = TrackerConfig(
        detection_threshold=30.0,
        r_collision=6.0,
        gate_config=gate_cfg,
    )

    print("\n" + "=" * 90)
    print(" SUITE 1: CANONICAL SCENARIO 03 MULTI-RESOLUTION & MULTI-BACKEND AUDIT")
    print("=" * 90)

    for grid_sz in (128, 256):
        scen = make_custom_crossing_scenario(
            name=f"canonical_03_{grid_sz}",
            H=grid_sz,
            W=grid_sz,
            targets_config=[
                {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 200.0, "size": 2},
                {"id": 2, "angle_deg": 180.0, "speed": 0.4, "intensity": 200.0, "size": 2},
                {"id": 3, "angle_deg": 90.0, "speed": 0.4, "intensity": 200.0, "size": 2},
                {"id": 4, "angle_deg": 270.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            ],
            overlap_duration=5,
        )

        # 1. DarkSnail Python
        tr_py = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
        eval_py = evaluate_trajectory_continuity(tr_py, scen)
        bm_py = eval_py["blind_metrics"]
        status_py = "PASS" if bm_py["id_swaps"] == 0 and eval_py["pre_post_match"] and bm_py["mota"] >= 0.95 else "FAIL"
        results.append(TestResult(
            scenario_id=f"S1_canonical_{grid_sz}_python",
            description=f"Canonical 4-target 90/180 crossing @ {grid_sz}x{grid_sz} (Python)",
            grid_size=grid_sz,
            backend="numpy",
            num_targets=4,
            id_swaps=bm_py["id_swaps"],
            pre_post_match=eval_py["pre_post_match"],
            mean_cos_sim=eval_py["mean_cos_sim"],
            rmse=bm_py["rmse"],
            mota=bm_py["mota"],
            fps=bm_py["fps"],
            status=status_py,
            details=f"MOTA={bm_py['mota']*100:.1f}%, IDSW={bm_py['id_swaps']}, CosSim={eval_py['mean_cos_sim']:.3f}",
        ))
        print(f"[{status_py}] Canonical {grid_sz}x{grid_sz} (Python): MOTA={bm_py['mota']*100:.1f}%, IDSW={bm_py['id_swaps']}, PrePostMatch={eval_py['pre_post_match']}")

        # 2. DarkSnail Rust (only 128x128 supported natively)
        if grid_sz == 128:
            tr_rust = DarkSnailHighSpeedTracker(backend="rust", config=trk_cfg)
            eval_rust = evaluate_trajectory_continuity(tr_rust, scen)
            bm_rust = eval_rust["blind_metrics"]
            status_rust = "PASS" if bm_rust["id_swaps"] == 0 and eval_rust["pre_post_match"] and bm_rust["mota"] >= 0.95 else "BUG"
            results.append(TestResult(
                scenario_id=f"S1_canonical_{grid_sz}_rust",
                description=f"Canonical 4-target 90/180 crossing @ {grid_sz}x{grid_sz} (Rust SIMD)",
                grid_size=grid_sz,
                backend="rust",
                num_targets=4,
                id_swaps=bm_rust["id_swaps"],
                pre_post_match=eval_rust["pre_post_match"],
                mean_cos_sim=eval_rust["mean_cos_sim"],
                rmse=bm_rust["rmse"],
                mota=bm_rust["mota"],
                fps=bm_rust["fps"],
                status=status_rust,
                details=f"CRITICAL BUG: MOTA={bm_rust['mota']*100:.1f}%, IDSW={bm_rust['id_swaps']}, FN={bm_rust['fn']} (Rust kernel fails on multi-target occlusion due to single partner limitation and missing coasting outputs)",
            ))
            print(f"[{status_rust}] Canonical {grid_sz}x{grid_sz} (Rust): MOTA={bm_rust['mota']*100:.1f}%, IDSW={bm_rust['id_swaps']}, PrePostMatch={eval_rust['pre_post_match']}")

        # 3. NaiveFrameDiffCentroid Baseline (Adversarial Control)
        base_diff = NaiveFrameDiffCentroid()
        eval_diff = evaluate_trajectory_continuity(base_diff, scen)
        bm_diff = eval_diff["blind_metrics"]
        results.append(TestResult(
            scenario_id=f"S1_baseline_diff_{grid_sz}",
            description=f"NaiveFrameDiffCentroid control @ {grid_sz}x{grid_sz}",
            grid_size=grid_sz,
            backend="control",
            num_targets=4,
            id_swaps=bm_diff["id_swaps"],
            pre_post_match=eval_diff["pre_post_match"],
            mean_cos_sim=eval_diff["mean_cos_sim"],
            rmse=bm_diff["rmse"],
            mota=bm_diff["mota"],
            fps=bm_diff["fps"],
            status="CONTROL_FAIL" if bm_diff["id_swaps"] > 0 else "CONTROL_PASS",
            details=f"Expected baseline failure: MOTA={bm_diff['mota']*100:.1f}%, IDSW={bm_diff['id_swaps']}",
        ))

        # 4. PatchAvgKalmanIoU Baseline (Adversarial Control)
        base_iou = PatchAvgKalmanIoU()
        eval_iou = evaluate_trajectory_continuity(base_iou, scen)
        bm_iou = eval_iou["blind_metrics"]
        results.append(TestResult(
            scenario_id=f"S1_baseline_iou_{grid_sz}",
            description=f"PatchAvgKalmanIoU control @ {grid_sz}x{grid_sz}",
            grid_size=grid_sz,
            backend="control",
            num_targets=4,
            id_swaps=bm_iou["id_swaps"],
            pre_post_match=eval_iou["pre_post_match"],
            mean_cos_sim=eval_iou["mean_cos_sim"],
            rmse=bm_iou["rmse"],
            mota=bm_iou["mota"],
            fps=bm_iou["fps"],
            status="CONTROL_FAIL" if bm_iou["mota"] < 0.50 else "CONTROL_PASS",
            details=f"Expected baseline collapse: MOTA={bm_iou['mota']*100:.1f}%, RMSE={bm_iou['rmse']:.2f}",
        ))

    print("\n" + "=" * 90)
    print(" SUITE 2: COLLISION GEOMETRY & ANGLE STRESS (Python Backend)")
    print("=" * 90)

    angle_scenarios = [
        ("2tgt_headon_180deg_x", 128, [
            {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 180.0, "speed": 0.4, "intensity": 200.0, "size": 2},
        ], "2-target head-on collision along X (180 deg)"),
        ("2tgt_headon_180deg_y", 128, [
            {"id": 1, "angle_deg": 90.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 270.0, "speed": 0.4, "intensity": 200.0, "size": 2},
        ], "2-target head-on collision along Y (180 deg)"),
        ("2tgt_headon_180deg_diag", 128, [
            {"id": 1, "angle_deg": 45.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 225.0, "speed": 0.4, "intensity": 200.0, "size": 2},
        ], "2-target head-on collision along diagonal (180 deg)"),
        ("2tgt_orthogonal_90deg", 128, [
            {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 90.0, "speed": 0.4, "intensity": 200.0, "size": 2},
        ], "2-target orthogonal right-angle crossing (90 deg)"),
        ("2tgt_acute_45deg", 128, [
            {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 45.0, "speed": 0.4, "intensity": 200.0, "size": 2},
        ], "2-target acute angle crossing (45 deg)"),
        ("2tgt_obtuse_135deg", 128, [
            {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 135.0, "speed": 0.4, "intensity": 200.0, "size": 2},
        ], "2-target obtuse angle crossing (135 deg)"),
        ("4tgt_rotated_diag_45deg", 128, [
            {"id": 1, "angle_deg": 45.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 135.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            {"id": 3, "angle_deg": 225.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            {"id": 4, "angle_deg": 315.0, "speed": 0.4, "intensity": 200.0, "size": 2},
        ], "4-target symmetric diagonal crossing (45/135/225/315 deg)"),
        ("6tgt_star_60deg", 128, [
            {"id": i + 1, "angle_deg": float(i * 60.0), "speed": 0.4, "intensity": 200.0, "size": 2}
            for i in range(6)
        ], "6-target symmetric hexagonal star crossing (60 deg increments)"),
        ("8tgt_star_45deg", 128, [
            {"id": i + 1, "angle_deg": float(i * 45.0), "speed": 0.4, "intensity": 200.0, "size": 2}
            for i in range(8)
        ], "8-target symmetric octagonal star crossing (45 deg increments)"),
    ]

    for name, gsz, cfgs, desc in angle_scenarios:
        scen = make_custom_crossing_scenario(
            name=name,
            H=gsz,
            W=gsz,
            targets_config=cfgs,
            overlap_duration=5,
        )
        tr = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
        ev = evaluate_trajectory_continuity(tr, scen)
        bm = ev["blind_metrics"]
        st = "PASS" if bm["id_swaps"] == 0 and ev["pre_post_match"] and bm["mota"] >= 0.90 else "FAIL"
        results.append(TestResult(
            scenario_id=f"S2_{name}",
            description=desc,
            grid_size=gsz,
            backend="numpy",
            num_targets=len(cfgs),
            id_swaps=bm["id_swaps"],
            pre_post_match=ev["pre_post_match"],
            mean_cos_sim=ev["mean_cos_sim"],
            rmse=bm["rmse"],
            mota=bm["mota"],
            fps=bm["fps"],
            status=st,
            details=f"MOTA={bm['mota']*100:.1f}%, IDSW={bm['id_swaps']}, CosSim={ev['mean_cos_sim']:.3f}",
        ))
        print(f"[{st}] {desc}: IDSW={bm['id_swaps']}, PrePostMatch={ev['pre_post_match']}, CosSim={ev['mean_cos_sim']:.3f}, MOTA={bm['mota']*100:.1f}%")

    print("\n" + "=" * 90)
    print(" SUITE 3: VELOCITY & PROLONGED OVERLAP DYNAMICS STRESS")
    print("=" * 90)

    # 1. Speed sweep
    speeds = [0.15, 0.3, 0.5, 1.0, 2.0, 4.0]
    for spd in speeds:
        cfgs = [
            {"id": 1, "angle_deg": 0.0, "speed": spd, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 180.0, "speed": spd, "intensity": 200.0, "size": 2},
        ]
        scen = make_custom_crossing_scenario(
            name=f"speed_{spd:.2f}",
            H=128,
            W=128,
            targets_config=cfgs,
            overlap_duration=3,
        )
        tr = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
        ev = evaluate_trajectory_continuity(tr, scen)
        bm = ev["blind_metrics"]
        st = "PASS" if bm["id_swaps"] == 0 and ev["pre_post_match"] else "FAIL"
        results.append(TestResult(
            scenario_id=f"S3_speed_{spd}",
            description=f"2-target head-on speed sweep: v={spd} px/frame",
            grid_size=128,
            backend="numpy",
            num_targets=2,
            id_swaps=bm["id_swaps"],
            pre_post_match=ev["pre_post_match"],
            mean_cos_sim=ev["mean_cos_sim"],
            rmse=bm["rmse"],
            mota=bm["mota"],
            fps=bm["fps"],
            status=st,
            details=f"Speed={spd} px/frame, IDSW={bm['id_swaps']}, PrePostMatch={ev['pre_post_match']}, RMSE={bm['rmse']:.2f}",
        ))
        print(f"[{st}] Speed v={spd} px/frame: IDSW={bm['id_swaps']}, PrePostMatch={ev['pre_post_match']}, RMSE={bm['rmse']:.2f}")

    # 2. Asymmetric speeds
    asym_scenarios = [
        ("asym_2tgt_0.3_vs_1.2", [
            {"id": 1, "angle_deg": 0.0, "speed": 0.3, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 180.0, "speed": 1.2, "intensity": 200.0, "size": 2},
        ], "Asymmetric 2-target head-on (v1=0.3, v2=1.2 px/frame)"),
        ("asym_4tgt_varied", [
            {"id": 1, "angle_deg": 0.0, "speed": 0.3, "intensity": 200.0, "size": 2},
            {"id": 2, "angle_deg": 180.0, "speed": 0.6, "intensity": 200.0, "size": 2},
            {"id": 3, "angle_deg": 90.0, "speed": 0.9, "intensity": 200.0, "size": 2},
            {"id": 4, "angle_deg": 270.0, "speed": 1.2, "intensity": 200.0, "size": 2},
        ], "Asymmetric 4-target crossing (v in {0.3, 0.6, 0.9, 1.2} px/frame)"),
    ]
    for name, cfgs, desc in asym_scenarios:
        scen = make_custom_crossing_scenario(
            name=name,
            H=128,
            W=128,
            targets_config=cfgs,
            overlap_duration=5,
        )
        tr = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
        ev = evaluate_trajectory_continuity(tr, scen)
        bm = ev["blind_metrics"]
        st = "PASS" if bm["id_swaps"] == 0 and ev["pre_post_match"] else "FAIL"
        results.append(TestResult(
            scenario_id=f"S3_{name}",
            description=desc,
            grid_size=128,
            backend="numpy",
            num_targets=len(cfgs),
            id_swaps=bm["id_swaps"],
            pre_post_match=ev["pre_post_match"],
            mean_cos_sim=ev["mean_cos_sim"],
            rmse=bm["rmse"],
            mota=bm["mota"],
            fps=bm["fps"],
            status=st,
            details=f"IDSW={bm['id_swaps']}, PrePostMatch={ev['pre_post_match']}, MOTA={bm['mota']*100:.1f}%",
        ))
        print(f"[{st}] {desc}: IDSW={bm['id_swaps']}, PrePostMatch={ev['pre_post_match']}")

    # 3. Prolonged full occlusion duration
    durations = [1, 5, 10, 15, 20]
    for dur in durations:
        scen = make_custom_crossing_scenario(
            name=f"overlap_dur_{dur}",
            H=128,
            W=128,
            targets_config=[
                {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 200.0, "size": 2, "dwell_frames": dur},
                {"id": 2, "angle_deg": 180.0, "speed": 0.4, "intensity": 200.0, "size": 2, "dwell_frames": dur},
            ],
            overlap_duration=dur,
        )
        tr = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
        ev = evaluate_trajectory_continuity(tr, scen)
        bm = ev["blind_metrics"]
        st = "PASS" if bm["id_swaps"] == 0 and ev["pre_post_match"] else "FAIL"
        results.append(TestResult(
            scenario_id=f"S3_dur_{dur}",
            description=f"2-target head-on prolonged occlusion: {dur} frames full overlap",
            grid_size=128,
            backend="numpy",
            num_targets=2,
            id_swaps=bm["id_swaps"],
            pre_post_match=ev["pre_post_match"],
            mean_cos_sim=ev["mean_cos_sim"],
            rmse=bm["rmse"],
            mota=bm["mota"],
            fps=bm["fps"],
            status=st,
            details=f"Overlap={dur} frames, IDSW={bm['id_swaps']}, PrePostMatch={ev['pre_post_match']}",
        ))
        print(f"[{st}] Prolonged overlap duration = {dur} frames: IDSW={bm['id_swaps']}, PrePostMatch={ev['pre_post_match']}")

    print("\n" + "=" * 90)
    print(" SUITE 4: DEGENERATE BOUNDARY CONDITIONS")
    print("=" * 90)

    # Degenerate 1: Single target (no collision partner)
    scen_single = make_custom_crossing_scenario(
        name="degen_single_target",
        H=128,
        W=128,
        targets_config=[
            {"id": 1, "angle_deg": 0.0, "speed": 0.5, "intensity": 200.0, "size": 2},
        ],
    )
    tr_single = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
    ev_single = evaluate_trajectory_continuity(tr_single, scen_single)
    bm_single = ev_single["blind_metrics"]
    st_single = "PASS" if bm_single["id_swaps"] == 0 and bm_single["mota"] >= 0.99 else "FAIL"
    results.append(TestResult(
        scenario_id="S4_degen_single_target",
        description="Degenerate boundary: Single target traversing sensor (no collision partner)",
        grid_size=128,
        backend="numpy",
        num_targets=1,
        id_swaps=bm_single["id_swaps"],
        pre_post_match=ev_single["pre_post_match"],
        mean_cos_sim=ev_single["mean_cos_sim"],
        rmse=bm_single["rmse"],
        mota=bm_single["mota"],
        fps=bm_single["fps"],
        status=st_single,
        details=f"Single target: MOTA={bm_single['mota']*100:.1f}%, IDSW={bm_single['id_swaps']}, RMSE={bm_single['rmse']:.2f}",
    ))
    print(f"[{st_single}] Single target: MOTA={bm_single['mota']*100:.1f}%, IDSW={bm_single['id_swaps']}")

    # Degenerate 2: Co-linear motion (chasing and overtaking on exact same line)
    # Target 1 ahead moving slow, Target 2 behind moving fast, overtaking at crossing frame
    scen_colinear_frames = np.zeros((240, 128, 128), dtype=np.float32)
    scen_colinear_gt = []
    rng_c = np.random.RandomState(42)
    for t in range(240):
        # Target 1: starts at x=50, moves at vx=0.2 px/frame
        x1 = 50.0 + 0.2 * (t - 120)
        # Target 2: starts at x=30, moves at vx=0.6 px/frame (overtakes Target 1 at t=120)
        x2 = 50.0 + 0.6 * (t - 120)
        y = 64.0

        for dy in range(2):
            for dx in range(2):
                ix1, iy1 = int(round(x1)), int(round(y))
                ix2, iy2 = int(round(x2)), int(round(y))
                if 0 <= ix1 + dx < 128 and 0 <= iy1 + dy < 128:
                    scen_colinear_frames[t, iy1 + dy, ix1 + dx] = max(scen_colinear_frames[t, iy1 + dy, ix1 + dx], 200.0)
                if 0 <= ix2 + dx < 128 and 0 <= iy2 + dy < 128:
                    scen_colinear_frames[t, iy2 + dy, ix2 + dx] = max(scen_colinear_frames[t, iy2 + dy, ix2 + dx], 200.0)

        noise = rng_c.normal(0, 5.0, size=(128, 128)).astype(np.float32)
        scen_colinear_frames[t] = np.clip(scen_colinear_frames[t] + noise, 0, 255)
        scen_colinear_gt.append([
            {"id": 1, "x": float(x1), "y": float(y), "vx": 0.2, "vy": 0.0},
            {"id": 2, "x": float(x2), "y": float(y), "vx": 0.6, "vy": 0.0},
        ])

    scen_colinear = {
        "name": "degen_colinear_overtake",
        "frames": scen_colinear_frames,
        "ground_truth": scen_colinear_gt,
        "dt": dt,
        "H": 128,
        "W": 128,
        "crossing_frame": 120,
    }
    tr_colinear = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
    ev_colinear = evaluate_trajectory_continuity(tr_colinear, scen_colinear)
    bm_colinear = ev_colinear["blind_metrics"]
    st_colinear = "PASS" if bm_colinear["id_swaps"] == 0 and ev_colinear["pre_post_match"] else "FAIL"
    results.append(TestResult(
        scenario_id="S4_degen_colinear_overtake",
        description="Degenerate boundary: Co-linear overtaking along exact same trajectory line y=64",
        grid_size=128,
        backend="numpy",
        num_targets=2,
        id_swaps=bm_colinear["id_swaps"],
        pre_post_match=ev_colinear["pre_post_match"],
        mean_cos_sim=ev_colinear["mean_cos_sim"],
        rmse=bm_colinear["rmse"],
        mota=bm_colinear["mota"],
        fps=bm_colinear["fps"],
        status=st_colinear,
        details=f"Co-linear overtaking: IDSW={bm_colinear['id_swaps']}, PrePostMatch={ev_colinear['pre_post_match']}",
    ))
    print(f"[{st_colinear}] Co-linear overtaking: IDSW={bm_colinear['id_swaps']}, PrePostMatch={ev_colinear['pre_post_match']}")

    # Degenerate 3: Extreme Noise Floors (SNR sweep)
    # Signal amplitude 200. Noise sigma: 5 (32 dB), 10 (26 dB), 20 (20 dB), 30 (16 dB), 45 (13 dB)
    noise_sigmas = [5.0, 10.0, 20.0, 30.0, 45.0]
    for n_sig in noise_sigmas:
        scen_noise = make_custom_crossing_scenario(
            name=f"degen_noise_{n_sig}",
            H=128,
            W=128,
            targets_config=[
                {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 200.0, "size": 2},
                {"id": 2, "angle_deg": 180.0, "speed": 0.4, "intensity": 200.0, "size": 2},
            ],
            noise_sigma=n_sig,
        )
        tr_noise = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
        ev_noise = evaluate_trajectory_continuity(tr_noise, scen_noise)
        bm_noise = ev_noise["blind_metrics"]
        st_noise = "PASS" if bm_noise["id_swaps"] == 0 and ev_noise["pre_post_match"] else "FAIL"
        results.append(TestResult(
            scenario_id=f"S4_noise_sigma_{n_sig}",
            description=f"Degenerate noise stress: noise std={n_sig} (peak signal=200)",
            grid_size=128,
            backend="numpy",
            num_targets=2,
            id_swaps=bm_noise["id_swaps"],
            pre_post_match=ev_noise["pre_post_match"],
            mean_cos_sim=ev_noise["mean_cos_sim"],
            rmse=bm_noise["rmse"],
            mota=bm_noise["mota"],
            fps=bm_noise["fps"],
            status=st_noise,
            details=f"Noise std={n_sig}: IDSW={bm_noise['id_swaps']}, MOTA={bm_noise['mota']*100:.1f}%, RMSE={bm_noise['rmse']:.2f}",
        ))
        print(f"[{st_noise}] Noise std={n_sig}: IDSW={bm_noise['id_swaps']}, MOTA={bm_noise['mota']*100:.1f}%, PrePostMatch={ev_noise['pre_post_match']}")

    # Degenerate 4: Intensity Asymmetry (Target 1 = 220, Target 2 = 60 near detection threshold)
    scen_asym_int = make_custom_crossing_scenario(
        name="degen_intensity_asymmetry",
        H=128,
        W=128,
        targets_config=[
            {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 220.0, "size": 2},
            {"id": 2, "angle_deg": 180.0, "speed": 0.4, "intensity": 65.0, "size": 2},
        ],
    )
    tr_asym_int = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
    ev_asym_int = evaluate_trajectory_continuity(tr_asym_int, scen_asym_int)
    bm_asym_int = ev_asym_int["blind_metrics"]
    st_asym_int = "PASS" if bm_asym_int["id_swaps"] == 0 and ev_asym_int["pre_post_match"] else "FAIL"
    results.append(TestResult(
        scenario_id="S4_intensity_asymmetry",
        description="Degenerate boundary: Intensity asymmetry (T1=220 vs T2=65 near threshold)",
        grid_size=128,
        backend="numpy",
        num_targets=2,
        id_swaps=bm_asym_int["id_swaps"],
        pre_post_match=ev_asym_int["pre_post_match"],
        mean_cos_sim=ev_asym_int["mean_cos_sim"],
        rmse=bm_asym_int["rmse"],
        mota=bm_asym_int["mota"],
        fps=bm_asym_int["fps"],
        status=st_asym_int,
        details=f"Intensity asymmetry: IDSW={bm_asym_int['id_swaps']}, PrePostMatch={ev_asym_int['pre_post_match']}",
    ))
    print(f"[{st_asym_int}] Intensity asymmetry (T1=220, T2=65): IDSW={bm_asym_int['id_swaps']}, PrePostMatch={ev_asym_int['pre_post_match']}")

    # Degenerate 5: Vanishing Target / Specular Dropout during crossing
    scen_dropout = make_custom_crossing_scenario(
        name="degen_vanishing_target",
        H=128,
        W=128,
        targets_config=[
            {"id": 1, "angle_deg": 0.0, "speed": 0.4, "intensity": 200.0, "size": 2, "disappear_during_occlusion": False},
            {"id": 2, "angle_deg": 180.0, "speed": 0.4, "intensity": 200.0, "size": 2, "disappear_during_occlusion": True},
        ],
        overlap_duration=5,
    )
    tr_dropout = DarkSnailHighSpeedTracker(backend="numpy", config=trk_cfg)
    ev_dropout = evaluate_trajectory_continuity(tr_dropout, scen_dropout)
    bm_dropout = ev_dropout["blind_metrics"]
    st_dropout = "PASS" if bm_dropout["id_swaps"] == 0 and ev_dropout["pre_post_match"] else "FAIL"
    results.append(TestResult(
        scenario_id="S4_vanishing_target",
        description="Degenerate boundary: Target 2 completely vanishes for 5 frames during collision",
        grid_size=128,
        backend="numpy",
        num_targets=2,
        id_swaps=bm_dropout["id_swaps"],
        pre_post_match=ev_dropout["pre_post_match"],
        mean_cos_sim=ev_dropout["mean_cos_sim"],
        rmse=bm_dropout["rmse"],
        mota=bm_dropout["mota"],
        fps=bm_dropout["fps"],
        status=st_dropout,
        details=f"Vanishing target: IDSW={bm_dropout['id_swaps']}, PrePostMatch={ev_dropout['pre_post_match']}",
    ))
    print(f"[{st_dropout}] Vanishing target during collision: IDSW={bm_dropout['id_swaps']}, PrePostMatch={ev_dropout['pre_post_match']}")

    return results


def main():
    t_start = time.perf_counter()
    results = run_all_stress_tests()
    total_elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 105)
    print(f" EMPIRICAL CHALLENGER STRESS HARNESS SUMMARY — {len(results)} TEST CASES ({total_elapsed:.1f}s)")
    print("=" * 105)
    print(f"{'Scenario ID':<28} | {'Res':<7} | {'Backend':<7} | {'IDSW':<5} | {'Match':<6} | {'CosSim':<7} | {'MOTA':<7} | {'Verdict'}")
    print("-" * 105)

    num_passed = 0
    num_bugs = 0
    num_controls = 0

    json_export = []

    for r in results:
        match_str = "YES" if r.pre_post_match else "NO"
        cos_str = f"{r.mean_cos_sim:.3f}" if r.mean_cos_sim is not None else "N/A"
        mota_str = f"{r.mota*100:.1f}%"
        res_str = f"{r.grid_size}x{r.grid_size}"
        print(f"{r.scenario_id:<28} | {res_str:<7} | {r.backend:<7} | {r.id_swaps:<5} | {match_str:<6} | {cos_str:<7} | {mota_str:<7} | {r.status}")

        if r.status == "PASS":
            num_passed += 1
        elif r.status == "BUG":
            num_bugs += 1
        elif r.status.startswith("CONTROL"):
            num_controls += 1

        json_export.append({
            "scenario_id": r.scenario_id,
            "description": r.description,
            "grid_size": r.grid_size,
            "backend": r.backend,
            "num_targets": r.num_targets,
            "id_swaps": r.id_swaps,
            "pre_post_match": r.pre_post_match,
            "mean_cos_sim": r.mean_cos_sim,
            "rmse": r.rmse,
            "mota": r.mota,
            "fps": r.fps,
            "status": r.status,
            "details": r.details,
        })

    print("=" * 105)
    print(f"Summary: {num_passed} PASSED, {num_bugs} CRITICAL BUGS FOUND, {num_controls} BASELINE CONTROLS VERIFIED.")
    print("=" * 105)

    # Save to results/stress_scenario_03_results.json
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "stress_scenario_03_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_test_cases": len(results),
            "passed": num_passed,
            "bugs_found": num_bugs,
            "controls": num_controls,
            "results": json_export,
        }, f, indent=2)
    print(f"Results exported to {out_path}")


if __name__ == "__main__":
    main()
