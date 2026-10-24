"""
Unit and Integration Tests for Zero-Weight Optical Terminal 6-DOF GNC (R4.2).

Verifies:
1. HCW relative orbit dynamics and unit quaternion attitude kinematics.
2. 3-point discrete Morse innovation gate: glint rejection (r_k <= 0.35) and step maneuver detection.
3. Synthetic optical camera projection and SO(3) rotation matrix orthogonality.
4. Autonomous terminal intercept from R0 = 500 m to docking miss distance <= 0.05 m with:
   - 0 neural weights
   - 0 track-ID swaps
   - Final docking contact velocity <= 0.02 m/s
   - Rejection of 1-sample optical flares under target evasive burns.
5. Integrity and validity of `stylo/results/stylo_6dof_gnc_summary.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pytest

from darksnail.gnc_6dof import (
    SpacecraftPlant6DOF,
    SpacecraftPlantConfig,
    SyntheticOpticalCamera,
    SyntheticCameraConfig,
    DiscreteMorseInnovationGate,
    ZeroWeightGNCGuidance,
    GNCGuidanceConfig,
    quaternion_norm,
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_to_rot_matrix,
    quaternion_from_two_vectors,
    CLASSIFICATION_NOMINAL,
    CLASSIFICATION_STEP_MANEUVER,
    CLASSIFICATION_GLINT_REJECTED,
    run_6dof_terminal_intercept,
)


class TestGNCPhysicsAndKinematics:
    """Verifies physical dynamics, quaternion algebra, and SO(3) rotational properties."""

    def test_quaternion_algebra_and_so3_orthogonality(self):
        """Verify quaternion norm, Hamilton product, and rotation matrix orthogonality."""
        q1 = quaternion_normalize(np.array([1.0, 2.0, -1.0, 0.5]))
        assert abs(quaternion_norm(q1) - 1.0) < 1e-14

        # Conjugate and identity inverse
        q1_inv = quaternion_conjugate(q1)
        q_ident = quaternion_multiply(q1, q1_inv)
        assert abs(q_ident[0] - 1.0) < 1e-14
        assert np.max(np.abs(q_ident[1:])) < 1e-14

        # SO(3) direction cosine matrix properties: R * R^T = I and det(R) = +1
        R = quaternion_to_rot_matrix(q1)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(float(np.linalg.det(R)) - 1.0) < 1e-12

    def test_hcw_dynamics_and_quaternion_propagation(self):
        """Propagate plant for 1,000 steps; verify quaternion norm preservation."""
        plant = SpacecraftPlant6DOF()
        plant.set_initial_state(
            rel_pos=[100.0, -200.0, 50.0],
            rel_vel=[0.05, -0.10, 0.02],
            att_quat=[1.0, 0.0, 0.0, 0.0],
            att_rate=[0.01, -0.02, 0.015],
        )

        dt = 0.05
        for _ in range(1000):
            thrust = np.array([2.0, -1.5, 0.5])
            torque = np.array([0.1, -0.05, 0.08])
            plant.step_rk4(dt, thrust, torque)

            # Strict S^3 unit-sphere norm preservation: | ||q|| - 1 | < 1e-12
            assert abs(quaternion_norm(plant.q) - 1.0) < 1e-12

        # Verify HCW continuous motion
        assert np.all(np.isfinite(plant.rel_state))
        assert np.all(np.isfinite(plant.omega))


class TestDiscreteMorseGateAndOpticalSensor:
    """Verifies sensor projection and discrete Morse innovation gate."""

    def test_camera_projection_and_los_reconstruction(self):
        """Test optical camera projection to pixels and reconstruction of line-of-sight."""
        camera = SyntheticOpticalCamera(SyntheticCameraConfig(pixel_noise_sigma=0.0, glint_probability=0.0))
        rel_pos = np.array([50.0, 25.0, -10.0])
        q_chaser = np.array([1.0, 0.0, 0.0, 0.0])

        meas = camera.measure(rel_pos, q_chaser, inject_glint=False)
        assert meas["is_glint"] is False
        assert abs(meas["range_true"] - float(np.linalg.norm(rel_pos))) < 1e-9

        # Reconstructed LOS unit vector in LVLH must match true normalized rel_pos
        los_true = rel_pos / np.linalg.norm(rel_pos)
        assert np.allclose(meas["los_lvlh"], los_true, atol=1e-4)

    def test_discrete_morse_gate_glint_rejection_vs_step_maneuver(self):
        """Verify return ratio r_k <= 0.35 rejects 1-sample noise glints while r_k > 0.35 detects steps."""
        gate = DiscreteMorseInnovationGate(max_return_ratio=0.35, jump_threshold=5.0)

        # 1. 1-Sample Glint Flare (pulse that returns to baseline)
        y_prev = np.array([0.0, 0.0])
        y_curr = np.array([20.0, 20.0])  # Glint spike
        y_next = np.array([0.0, 0.0])    # Returns immediately to baseline
        cls_glint, ratio_glint = gate.filter_innovation(y_prev, y_curr, y_next)

        assert cls_glint == CLASSIFICATION_GLINT_REJECTED
        assert ratio_glint <= 0.35

        # 2. Sustained Impulsive Step Maneuver (burn that persists)
        y_prev = np.array([0.0, 0.0])
        y_curr = np.array([15.0, 15.0])  # Step burn
        y_next = np.array([15.2, 14.8])  # Persists at step level
        cls_step, ratio_step = gate.filter_innovation(y_prev, y_curr, y_next)

        assert cls_step == CLASSIFICATION_STEP_MANEUVER
        assert ratio_step > 0.35


class TestAutonomousClosedLoopTerminalIntercept:
    """End-to-end integration test from R0 = 500 m to docking miss distance <= 0.05 m."""

    def test_closed_loop_terminal_intercept_500m_to_5cm(self, tmp_path):
        """Execute full flight simulation and verify all acceptance criteria."""
        summary = run_6dof_terminal_intercept(
            initial_range=500.0,
            dt=0.05,
            max_duration=180.0,
            target_impulse_time=15.0,
            target_impulse_dv=[0.0, 0.25, 0.0],
            output_dir=tmp_path,
        )

        # 1. Successful autonomous intercept
        assert summary["intercept_success"] is True, "Terminal intercept must succeed"

        # 2. Docking contact miss distance <= 0.05 m (5 cm)
        assert summary["final_miss_distance_m"] <= 0.05, (
            f"Expected miss distance <= 0.05 m, got {summary['final_miss_distance_m']}"
        )

        # 3. Docking contact velocity <= 0.02 m/s (2 cm/s)
        assert summary["final_docking_speed_m_s"] <= 0.02, (
            f"Expected docking contact velocity <= 0.02 m/s, got {summary['final_docking_speed_m_s']}"
        )

        # 4. Strictly ZERO neural weights
        assert summary["neural_weights"] == 0, "Guidance must use strictly 0 neural weights"

        # 5. Strictly ZERO track-ID swaps
        assert summary["track_id_swaps"] == 0, "Optical tracker must have 0 track-ID swaps"

        # 6. Rejection of optical glint flares
        assert summary["glints_rejected"] > 0, "Must detect and reject optical glints"

        # 7. Target evasive maneuver handled
        assert summary["maneuvers_detected"] > 0

        # 8. S^3 quaternion norm residual < 1e-12
        assert summary["quaternion_norm_residual"] < 1e-12

    def test_stylo_6dof_gnc_summary_json_artifact(self):
        """Verify the official result artifact `stylo/results/stylo_6dof_gnc_summary.json`."""
        summary_file = Path(__file__).resolve().parent.parent / "results" / "stylo_6dof_gnc_summary.json"
        assert summary_file.exists(), f"Summary file missing: {summary_file}"

        data = json.loads(summary_file.read_text(encoding="utf-8"))
        assert data["intercept_success"] is True
        assert data["final_miss_distance_m"] <= 0.05
        assert data["final_docking_speed_m_s"] <= 0.02
        assert data["track_id_swaps"] == 0
        assert data["neural_weights"] == 0
        assert data["glints_rejected"] > 0
        assert data["quaternion_norm_residual"] < 1e-12
