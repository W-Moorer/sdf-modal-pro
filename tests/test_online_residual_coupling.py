from __future__ import annotations

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.modal_ilc import (
    ActivePatchSet,
    OnlineReducedDynamicState,
    ReducedState,
    baseline_reduced_newmark_step,
    build_normal_patch_load_bases,
    build_patch_residual_basis,
    solve_online_residual_dynamic_step,
    solve_online_residual_contact_step,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def test_online_gap_query_uses_residual_deformation_and_changes_contact_force() -> None:
    case = _phase6_case()
    state = ReducedState(r=np.zeros(3), theta=np.zeros(3), eta_k=np.zeros(case["retained_modes"].shape[1]))
    points, areas = _distributed_patch_contact_points(case, case["right_patches"][:1], penetration=0.005)

    residual = solve_online_residual_contact_step(
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
        max_iterations=12,
        tolerance=1.0e-10,
        include_residual=True,
    )
    no_residual = solve_online_residual_contact_step(
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
        max_iterations=3,
        tolerance=1.0e-10,
        include_residual=False,
    )

    assert len(residual.iterations) > 2
    assert residual.iterations[0].residual_deformation_norm == 0.0
    assert residual.iterations[1].residual_deformation_norm > 0.0
    assert residual.iterations[1].min_gap != residual.iterations[0].min_gap
    assert np.linalg.norm(residual.residual_deformation) > 0.0
    assert abs(residual.normal_force - no_residual.normal_force) / no_residual.normal_force > 1.0e-4


def test_online_no_contact_regresses_to_modal_flexible_body() -> None:
    case = _phase6_case()
    eta = np.array([1.0e-5, -2.0e-5, 1.5e-5])
    state = ReducedState(r=np.zeros(3), theta=np.zeros(3), eta_k=eta)
    patch = case["right_patches"][0]
    points = (patch.centroid + 0.5 * patch.normal)[None, :]
    areas = np.array([patch.area])

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
        max_iterations=8,
        tolerance=1.0e-10,
    )

    modal = case["retained_modes"] @ eta
    assert result.active_set.is_empty
    assert result.samples.sample_count == 0
    assert result.normal_force == 0.0
    np.testing.assert_allclose(result.residual_deformation, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.displacement, modal, rtol=0.0, atol=1.0e-12)


def test_online_dynamic_step_solves_eta_correction_with_residual_contact() -> None:
    case = _phase6_case()
    previous = OnlineReducedDynamicState(
        state=ReducedState(r=np.zeros(3), theta=np.zeros(3), eta_k=np.zeros(case["retained_modes"].shape[1])),
        eta_velocity=np.zeros(case["retained_modes"].shape[1]),
        eta_acceleration=np.zeros(case["retained_modes"].shape[1]),
        active_set=ActivePatchSet.empty(),
    )
    points, areas = _distributed_patch_contact_points(case, case["right_patches"][:1], penetration=0.005)

    result = solve_online_residual_dynamic_step(
        case["fem"].K,
        case["fem"].M,
        case["fem"].mesh,
        case["surface"],
        case["level"],
        case["retained_modes"],
        previous,
        case["residual_blocks"],
        case["load_bases"],
        points,
        penalty=20.0,
        dt=0.01,
        sample_areas=areas,
        max_iterations=12,
        tolerance=1.0e-4,
    )

    assert result.converged
    assert result.dynamic_state_dimension == previous.state.dynamic_dimension
    assert result.next.active_set.dynamic_dimension_contribution == 0
    assert result.iteration_residual_reduction > 1.0e3
    assert result.final_iteration_residual < 1.0e-4
    assert np.linalg.norm(result.next.state.eta_k - previous.state.eta_k) > 0.0
    assert np.linalg.norm(result.residual_deformation) > 0.0


def test_online_dynamic_no_contact_matches_baseline_newmark() -> None:
    case = _phase6_case()
    eta = np.array([1.0e-5, -2.0e-5, 1.5e-5])
    velocity = np.array([2.0e-5, -1.0e-5, 0.5e-5])
    acceleration = np.array([0.0, 1.0e-5, -1.0e-5])
    previous = OnlineReducedDynamicState(
        state=ReducedState(r=np.zeros(3), theta=np.zeros(3), eta_k=eta),
        eta_velocity=velocity,
        eta_acceleration=acceleration,
        active_set=ActivePatchSet.empty(),
    )
    patch = case["right_patches"][0]
    points = (patch.centroid + 0.5 * patch.normal)[None, :]
    areas = np.array([patch.area])

    coupled = solve_online_residual_dynamic_step(
        case["fem"].K,
        case["fem"].M,
        case["fem"].mesh,
        case["surface"],
        case["level"],
        case["retained_modes"],
        previous,
        case["residual_blocks"],
        case["load_bases"],
        points,
        penalty=20.0,
        dt=0.01,
        sample_areas=areas,
        max_iterations=8,
        tolerance=1.0e-10,
    )
    baseline = baseline_reduced_newmark_step(
        case["fem"].K,
        case["fem"].M,
        case["retained_modes"],
        previous,
        dt=0.01,
    )

    assert coupled.next.active_set.is_empty
    np.testing.assert_allclose(coupled.next.state.eta_k, baseline.state.eta_k, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(coupled.next.eta_velocity, baseline.eta_velocity, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(coupled.next.eta_acceleration, baseline.eta_acceleration, atol=1.0e-12, rtol=0.0)


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
