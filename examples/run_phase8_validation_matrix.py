from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.validation import run_phase8_validation_matrix, write_phase8_validation_matrix


def main() -> None:
    result = run_phase8_validation_matrix()
    output_dir = Path("outputs/phase8_validation_matrix")
    write_phase8_validation_matrix(result, output_dir)
    print(f"gate passed: {int(result.gate_passed)}")
    print(f"static max completeness error: {result.summary['static_max_completeness_error']:.6e}")
    print(
        "compressed residual max compliance error: "
        f"{result.summary['static_compressed_residual_max_compliance_error']:.6e}"
    )
    print(f"quasi-static low-mode force error: {result.summary['quasi_static_low_mode_force_error']:.6e}")
    print(f"quasi-static residual ILC force error: {result.summary['quasi_static_residual_force_error']:.6e}")
    print(f"moving contact force jump: {result.summary['moving_contact_force_jump']:.6e}")
    print(f"moving contact gap jump: {result.summary['moving_contact_gap_jump']:.6e}")
    print(f"small dynamic peak force error: {result.summary['small_dynamic_peak_force_error']:.6e}")
    print(f"small dynamic impulse error: {result.summary['small_dynamic_impulse_error']:.6e}")
    print(f"small dynamic estimated speedup: {result.summary['small_dynamic_estimated_runtime_speedup']:.6e}")
    print(f"large active patch fraction: {result.summary['large_active_patch_fraction']:.6e}")
    print(f"wrote {output_dir / 'tables' / 'phase8_summary.csv'}")
    print(f"wrote {output_dir / 'logs' / 'phase8_validation_summary.md'}")
    print(f"wrote {output_dir / 'figures' / 'force_displacement_curve.png'}")
    print(f"wrote {output_dir / 'figures' / 'active_patch_history.png'}")


if __name__ == "__main__":
    main()
