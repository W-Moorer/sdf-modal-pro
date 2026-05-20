from __future__ import annotations

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.modal_ilc import (
    ReducedState,
    build_normal_patch_load_bases,
    build_patch_residual_basis,
    solve_online_residual_contact_step,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def test_contact_alpha_fixed_point_iteration_converges_and_updates_alpha() -> None:
    case = _phase6_case()
    state = ReducedState(r=np.zeros(3), theta=np.zeros(3), eta_k=np.zeros(case["retained_modes"].shape[1]))
    points, areas = _distributed_patch_contact_points(case, case["right_patches"][:1], penetration=0.005)

    result = solve_online_residual_contact_step(
        case["fem"].mesh,
        case["surface"],
        case["level"],
        case["retained_modes"],
        state,
        case["residual_blocks"],
        case["load_bases"],
        points,
        penalty=20.0,
        sample_areas=areas,
        max_iterations=16,
        tolerance=1.0e-10,
    )
    residuals = np.asarray([iteration.alpha_update_norm for iteration in result.iterations], dtype=float)
    alpha_values = [iteration.alpha.copy() for iteration in result.iterations if iteration.alpha.size]

    assert result.converged
    assert result.final_alpha_residual < 1.0e-6
    assert result.alpha_residual_reduction > 1.0e3
    assert residuals.size >= 4
    assert np.all(np.diff(residuals) <= 1.0e-12)
    assert any(np.linalg.norm(alpha_values[index + 1] - alpha_values[index]) > 0.0 for index in range(len(alpha_values) - 1))


def _phase6_case():
    fem = cantilever_block(nx=3, ny=2, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, level.patches)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=3)
    residual_basis = build_patch_residual_basis(fem.K, load_bases, low.modes, free_dofs=fem.free_dofs)
    right_patches = sorted(
        [patch for patch in level.patches if patch.key[0] == 0 and patch.key[1] == 1],
        key=lambda patch: (patch.centroid[1], patch.centroid[2]),
    )
    return {
        "fem": fem,
        "surface": surface,
        "level": level,
        "load_bases": load_bases,
        "retained_modes": low.modes,
        "residual_blocks": {block.patch_id: block for block in residual_basis.blocks},
        "right_patches": right_patches,
    }


def _distributed_patch_contact_points(case, patches, penetration: float):
    points = []
    areas = []
    for patch in patches:
        for triangle_id in patch.triangle_indices:
            points.append(case["surface"].centroids[triangle_id] - penetration * case["surface"].normals[triangle_id])
            areas.append(case["surface"].areas[triangle_id])
    return np.asarray(points, dtype=float), np.asarray(areas, dtype=float)
