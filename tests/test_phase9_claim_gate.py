from __future__ import annotations

from modal_contact_rom.validation import run_phase9_claim_gate, write_phase9_claim_gate


def test_phase9_ablation_claim_gate_meets_aim_outputs(tmp_path) -> None:
    result = run_phase9_claim_gate()

    assert result.gate_passed
    assert {row["ablation"] for row in result.ablation_rows} == {"A", "B", "C", "D", "E"}
    assert all(int(row["passed"]) == 1 for row in result.claim_rows)
    assert result.summary["sdf_projection_cases_passed"] == result.summary["sdf_projection_cases"]
    assert result.summary["three_way_external_cases_passed"] == result.summary["three_way_external_cases"]
    assert result.summary["multi_scale_projection_error"] <= result.summary["single_scale_projection_error"]
    assert result.summary["multi_scale_patch_alpha_dofs"] < result.summary["node_level_contact_dofs"]

    write_phase9_claim_gate(result, tmp_path)

    expected_tables = [
        "static_completeness.csv",
        "compliance_error.csv",
        "contact_force_error.csv",
        "runtime_scaling.csv",
        "ablation_summary.csv",
        "sdf_patch_projection.csv",
        "three_way_external_fem.csv",
        "claim_gate.csv",
    ]
    for filename in expected_tables:
        assert (tmp_path / "tables" / filename).is_file()

    expected_figures = [
        "method_pipeline.png",
        "patch_hierarchy.png",
        "force_displacement_curve.png",
        "active_patch_history.png",
        "runtime_scaling.png",
        "sdf_patch_projection_map.png",
    ]
    for filename in expected_figures:
        assert (tmp_path / "figures" / filename).is_file()

    claims_text = (tmp_path / "logs" / "patch_ilc_claims.md").read_text(encoding="utf-8")
    assert "Phase 9 gate passed: 1" in claims_text
    assert "Claim 7 [PASS]" in claims_text
