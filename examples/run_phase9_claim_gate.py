from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.validation import run_phase9_claim_gate, write_phase9_claim_gate


def main() -> None:
    result = run_phase9_claim_gate()
    output_dir = Path("results/paper_patch_ilc")
    write_phase9_claim_gate(result, output_dir)

    print(f"phase9 gate passed: {int(result.gate_passed)}")
    print(f"claims passed: {result.summary['phase9_claims_passed']} / {result.summary['phase9_claim_count']}")
    print(f"single-scale projection error: {result.summary['single_scale_projection_error']:.6e}")
    print(f"multi-scale projection error: {result.summary['multi_scale_projection_error']:.6e}")
    print(f"node-level contact dofs: {result.summary['node_level_contact_dofs']}")
    print(f"multi-scale patch alpha dofs: {result.summary['multi_scale_patch_alpha_dofs']}")
    print(
        "sdf projection evidence cases: "
        f"{result.summary['sdf_projection_cases_passed']} / {result.summary['sdf_projection_cases']}"
    )
    print(
        "complex surface cases: "
        f"{result.summary['complex_surface_cases_passed']} / {result.summary['complex_surface_cases']}"
    )
    print(
        "performance optimization cases: "
        f"{result.summary['performance_optimization_cases_passed']} / {result.summary['performance_optimization_cases']}"
    )
    print(
        "three-way external FEM cases: "
        f"{result.summary['three_way_external_cases_passed']} / {result.summary['three_way_external_cases']}"
    )
    print(f"wrote {output_dir / 'tables' / 'ablation_summary.csv'}")
    print(f"wrote {output_dir / 'tables' / 'sdf_patch_projection.csv'}")
    print(f"wrote {output_dir / 'tables' / 'complex_surface_moving_contact.csv'}")
    print(f"wrote {output_dir / 'tables' / 'performance_optimization.csv'}")
    print(f"wrote {output_dir / 'tables' / 'three_way_external_fem.csv'}")
    print(f"wrote {output_dir / 'tables' / 'claim_gate.csv'}")
    print(f"wrote {output_dir / 'figures' / 'method_pipeline.png'}")
    print(f"wrote {output_dir / 'figures' / 'patch_hierarchy.png'}")
    print(f"wrote {output_dir / 'figures' / 'force_displacement_curve.png'}")
    print(f"wrote {output_dir / 'figures' / 'active_patch_history.png'}")
    print(f"wrote {output_dir / 'figures' / 'runtime_scaling.png'}")
    print(f"wrote {output_dir / 'figures' / 'sdf_patch_projection_map.png'}")
    print(f"wrote {output_dir / 'figures' / 'complex_surface_moving_contact.png'}")
    print(f"wrote {output_dir / 'figures' / 'performance_optimization.png'}")
    print(f"wrote {output_dir / 'logs' / 'patch_ilc_claims.md'}")


if __name__ == "__main__":
    main()
