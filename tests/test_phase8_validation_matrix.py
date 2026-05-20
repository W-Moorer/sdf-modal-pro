from __future__ import annotations

from modal_contact_rom.validation import run_phase8_validation_matrix, write_phase8_validation_matrix


def test_phase8_validation_matrix_meets_aim_gates_and_writes_outputs(tmp_path) -> None:
    result = run_phase8_validation_matrix()

    assert result.gate_passed
    assert result.summary["static_max_completeness_error"] < 1.0e-6
    assert result.summary["static_compressed_residual_max_compliance_error"] < 3.0e-2
    assert result.summary["quasi_static_residual_force_error"] < result.summary["quasi_static_low_mode_force_error"]
    assert result.summary["quasi_static_residual_dynamic_dof_ratio_vs_ordinary"] < 1.0
    assert result.summary["moving_contact_force_jump"] < 5.0e-2
    assert result.summary["moving_contact_gap_jump"] < 5.0e-2
    assert result.summary["moving_contact_energy_drift_excess_ratio"] <= 1.0
    assert result.summary["small_dynamic_displacement_rmse"] >= 0.0
    assert result.summary["small_dynamic_velocity_rmse"] >= 0.0
    assert result.summary["small_dynamic_peak_force_error"] < 1.0e-1
    assert result.summary["small_dynamic_impulse_error"] < 1.0e-1
    assert result.summary["small_dynamic_measured_wall_runtime_ratio"] > 0.0
    assert result.summary["small_dynamic_solver_dimension_speedup"] > 5.0
    assert result.summary["large_active_patch_fraction"] < 0.35
    assert result.summary["large_dae_dimension_is_contact_independent"] == 1

    write_phase8_validation_matrix(result, tmp_path)

    assert (tmp_path / "tables" / "static_completeness.csv").read_text(encoding="utf-8").startswith("patch_id")
    assert "low_modes_plus_patch_residual_ilc" in (
        tmp_path / "tables" / "quasi_static_indentation.csv"
    ).read_text(encoding="utf-8")
    assert "force_jump" in (tmp_path / "tables" / "moving_contact.csv").read_text(encoding="utf-8")
    assert "solver_dimension_speedup" in (
        tmp_path / "tables" / "small_dynamic_full_fem.csv"
    ).read_text(encoding="utf-8")
    assert "patch_residual_ilc" in (tmp_path / "tables" / "large_performance.csv").read_text(encoding="utf-8")
    assert "multi_scale_patch_residual_ilc" in (tmp_path / "tables" / "large_performance.csv").read_text(
        encoding="utf-8"
    )
    assert "Phase 8 Validation Matrix Summary" in (
        tmp_path / "logs" / "phase8_validation_summary.md"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "figures" / "force_displacement_curve.png").is_file()
    assert (tmp_path / "figures" / "active_patch_history.png").is_file()
