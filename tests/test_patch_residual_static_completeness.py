from __future__ import annotations

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.modal_ilc import (
    ActivePatchSet,
    build_normal_patch_load_bases,
    build_patch_residual_basis,
    compliance_errors,
    recover_residual_deformation,
    static_completeness_errors,
    write_patch_residual_tables,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def test_patch_residual_modes_restore_full_static_attachment_solution() -> None:
    fem, load_bases, retained_modes = _phase4_case(num_modes=3)
    residual_basis = build_patch_residual_basis(fem.K, load_bases, retained_modes, free_dofs=fem.free_dofs)
    alpha = np.linspace(0.2, 1.4, residual_basis.alpha_dimension)

    u_static = residual_basis.static_response(alpha)
    u_rec = residual_basis.reconstructed_response(alpha)
    relative_error = np.linalg.norm(u_rec - u_static) / np.linalg.norm(u_static)

    assert relative_error < 1.0e-10
    assert float(np.max(static_completeness_errors(residual_basis))) < 1.0e-10
    np.testing.assert_allclose(u_static[fem.fixed_dofs], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(u_rec[fem.fixed_dofs], 0.0, atol=1.0e-14)


def test_patch_residual_modes_match_patch_compliance_and_improve_low_modes() -> None:
    fem, load_bases, retained_modes = _phase4_case(num_modes=2)
    residual_basis = build_patch_residual_basis(fem.K, load_bases, retained_modes, free_dofs=fem.free_dofs)

    low_error, residual_error = compliance_errors(residual_basis)

    assert float(np.max(residual_error)) < 1.0e-10
    assert float(np.mean(low_error)) > 1.0e-3
    assert float(np.mean(residual_error)) < 1.0e-6 * float(np.mean(low_error))


def test_patch_residual_blocks_feed_recovery_layer_without_expanding_state() -> None:
    fem, load_bases, retained_modes = _phase4_case(num_modes=3)
    residual_basis = build_patch_residual_basis(fem.K, load_bases, retained_modes, free_dofs=fem.free_dofs)
    alpha = np.linspace(1.0, 2.0, residual_basis.alpha_dimension)
    active = ActivePatchSet(
        active_patch_ids=tuple(basis.patch_id for basis in load_bases),
        active_load_basis_indices=tuple(range(residual_basis.alpha_dimension)),
        alpha=alpha,
    )

    residual = recover_residual_deformation(
        {block.patch_id: block for block in residual_basis.blocks},
        active,
        n_dofs=fem.mesh.n_dofs,
    )
    reconstructed = residual_basis.modal_static_response(alpha) + residual

    np.testing.assert_allclose(residual, residual_basis.residual_response(alpha), atol=1.0e-12)
    np.testing.assert_allclose(reconstructed, residual_basis.static_response(alpha), rtol=1.0e-11, atol=1.0e-12)
    assert active.dynamic_dimension_contribution == 0


def test_patch_residual_diagnostics_are_written(tmp_path) -> None:
    fem, load_bases, retained_modes = _phase4_case(num_modes=2)
    residual_basis = build_patch_residual_basis(fem.K, load_bases, retained_modes, free_dofs=fem.free_dofs)

    write_patch_residual_tables(residual_basis, tmp_path)

    assert (tmp_path / "residual_static_error.csv").read_text(encoding="utf-8").startswith("patch_id,basis_column")
    assert "low_relative_error" in (tmp_path / "patch_compliance_error.csv").read_text(encoding="utf-8")
    assert "compression_ratio" in (tmp_path / "compression_error.csv").read_text(encoding="utf-8")
    assert "Patch Residual Summary" in (tmp_path / "residual_summary.md").read_text(encoding="utf-8")


def _phase4_case(num_modes: int):
    fem = cantilever_block(nx=3, ny=2, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    patches = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium").patches
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, patches)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=num_modes)
    return fem, load_bases, low.modes
