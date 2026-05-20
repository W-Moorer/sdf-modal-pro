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


def test_online_contact_ilc_does_not_resize_dynamic_state() -> None:
    case = _phase6_case()
    state = ReducedState(r=np.zeros(3), theta=np.zeros(3), eta_k=np.zeros(case["retained_modes"].shape[1]))
    base_dimension = state.dynamic_dimension

    for points, areas in (
        _no_contact_points(case),
        _distributed_patch_contact_points(case, case["right_patches"][:1], penetration=0.005),
        _distributed_patch_contact_points(case, case["right_patches"][:2], penetration=0.005),
    ):
        result = solve_online_residual_contact_step(
            case["fem"].mesh,
            case["surface"],
            case["level"],
            case["retained_modes"],
            state,
            case["residual_blocks"],
            case["load_bases"],
            points,
            penalty=10.0,
            sample_areas=areas,
            max_iterations=12,
            tolerance=1.0e-10,
        )

        assert result.dynamic_state_dimension == base_dimension
        assert result.state.as_vector().size == base_dimension
        assert result.active_set.dynamic_dimension_contribution == 0
        assert not hasattr(result.state, "alpha")
        assert not hasattr(result.state, "q_patch")


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


def _no_contact_points(case):
    patch = case["right_patches"][0]
    return (patch.centroid + 0.25 * patch.normal)[None, :], np.array([patch.area])
