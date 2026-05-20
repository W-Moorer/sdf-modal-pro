from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.modal_ilc import (
    build_normal_patch_load_bases,
    build_patch_residual_basis,
    compliance_errors,
    static_completeness_errors,
    write_patch_residual_tables,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def main() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    medium = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, medium.patches)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=4)
    residual_basis = build_patch_residual_basis(fem.K, load_bases, low.modes, free_dofs=fem.free_dofs)
    output_dir = Path("outputs/patch_residual")
    write_patch_residual_tables(residual_basis, output_dir)

    low_error, residual_error = compliance_errors(residual_basis)
    static_errors = static_completeness_errors(residual_basis)
    print(f"patch residual columns: {residual_basis.alpha_dimension}")
    print(f"max static completeness error: {float(static_errors.max()):.6e}")
    print(f"mean low-mode compliance error: {float(low_error.mean()):.6e}")
    print(f"max residual compliance error: {float(residual_error.max()):.6e}")
    print(f"wrote {output_dir / 'residual_static_error.csv'}")
    print(f"wrote {output_dir / 'patch_compliance_error.csv'}")
    print(f"wrote {output_dir / 'compression_error.csv'}")
    print(f"wrote {output_dir / 'residual_summary.md'}")


if __name__ == "__main__":
    main()
