from __future__ import annotations

import numpy as np

from modal_contact_rom.contact_dynamics import (
    AdaptiveModalContactSimulator,
    FullFEMContactSimulator,
    PrescribedRigidPlane,
    PrescribedRigidSphere,
    patch_mode_dict,
)
from modal_contact_rom.contact_modes import build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.reduced_dynamics import compute_craig_bampton_basis, mass_orthonormalize
from modal_contact_rom.surface_patch import SurfacePatch, extract_surface, partition_surface_patches


def test_craig_bampton_interface_modes_preserve_interface_dofs() -> None:
    fem = cantilever_block(nx=3, ny=1, nz=1)
    interface_nodes = np.flatnonzero(np.isclose(fem.mesh.nodes[:, 0], fem.mesh.nodes[:, 0].max()))
    interface_dofs = fem.mesh.dof_indices(interface_nodes)

    cb = compute_craig_bampton_basis(
        fem.K,
        fem.M,
        fem.free_dofs,
        interface_dofs,
        num_fixed_interface_modes=3,
    )

    assert cb.constraint_modes.shape[1] == cb.interface_dofs.size
    assert cb.fixed_interface_modes.shape[1] == 3
    np.testing.assert_allclose(cb.constraint_modes[cb.interface_dofs, :], np.eye(cb.interface_dofs.size), atol=1.0e-12)
    np.testing.assert_allclose(cb.fixed_interface_modes[cb.interface_dofs, :], 0.0, atol=1.0e-12)


def test_contact_dynamics_activates_patch_near_rigid_contact() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(2, 2))
    loads = build_patch_normal_loads(fem.mesh, surface, patches)
    contact = solve_patch_contact_modes(fem.K, loads, fem.free_dofs)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=6)

    interface_nodes = np.flatnonzero(np.isclose(fem.mesh.nodes[:, 0], fem.mesh.nodes[:, 0].max()))
    cb = compute_craig_bampton_basis(
        fem.K,
        fem.M,
        fem.free_dofs,
        fem.mesh.dof_indices(interface_nodes),
        num_fixed_interface_modes=4,
    )
    base_basis = mass_orthonormalize(fem.M, np.column_stack((cb.basis, low.modes)))
    target_patch = _right_side_patch(patches)
    radius = 1.5
    center = target_patch.centroid + target_patch.normal * (radius - 0.05)

    simulator = AdaptiveModalContactSimulator(
        fem=fem,
        surface=surface,
        patches=patches,
        base_basis=base_basis,
        patch_modes=patch_mode_dict(contact.patch_ids, contact.modes),
        rigid_sphere=PrescribedRigidSphere(radius=radius, center=center),
        penalty=250.0,
        contact_damping=1.5,
        rayleigh_alpha=0.05,
        rayleigh_beta=0.002,
        activation_count=2,
    )
    result = simulator.run(dt=0.01, steps=40)

    active_ids = {patch_id for active in result.active_patch_history for patch_id in active}
    assert target_patch.patch_id in active_ids
    assert max(step.normal_force for step in result.steps) > 0.0
    assert min(step.min_gap for step in result.steps) < 0.0
    assert all(np.isfinite(step.elastic_energy + step.kinetic_energy) for step in result.steps)
    assert all(np.isfinite(step.energy_balance_error) for step in result.steps)


def test_full_fem_plane_contact_accepts_time_dependent_external_force() -> None:
    fem = cantilever_block(nx=2, ny=1, nz=1, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    top_nodes = np.flatnonzero(np.isclose(fem.mesh.nodes[:, 2], fem.mesh.nodes[:, 2].max()))
    free_top_nodes = np.setdiff1d(top_nodes, fem.fixed_nodes, assume_unique=False)
    force = np.zeros(fem.mesh.n_dofs, dtype=float)
    force[3 * free_top_nodes + 2] = 5.0

    simulator = FullFEMContactSimulator(
        fem=fem,
        surface=surface,
        rigid_sphere=PrescribedRigidPlane(point=np.array([0.0, 0.0, 0.51]), normal=np.array([0.0, 0.0, 1.0])),
        penalty=200.0,
        contact_damping=0.5,
        external_force=lambda time: force * min(time / 0.1, 1.0),
    )
    result = simulator.run(dt=0.005, steps=80)

    assert max(step.normal_force for step in result.steps) > 0.0
    assert max(step.max_penetration for step in result.steps) > 0.0
    assert result.steps[-1].external_work > 0.0
    assert all(np.isfinite(step.energy_balance_error) for step in result.steps)


def _right_side_patch(patches: list[SurfacePatch]) -> SurfacePatch:
    right = [patch for patch in patches if patch.key[0] == 0 and patch.key[1] == 1]
    assert right
    return max(right, key=lambda patch: patch.centroid[1] + patch.centroid[2])
