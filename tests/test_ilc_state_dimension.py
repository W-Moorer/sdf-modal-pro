from __future__ import annotations

import numpy as np

from modal_contact_rom.modal_ilc import (
    ActivePatchSet,
    PatchLoadBasis,
    PatchResidualBlock,
    ReducedState,
    project_contact_force_to_ilc,
    recover_residual_deformation,
)


def test_ilc_alpha_does_not_change_dynamic_state_dimension() -> None:
    state = ReducedState(
        r=np.zeros(3),
        theta=np.zeros(3),
        eta_k=np.zeros(5),
        lambda_=np.zeros(2),
    )
    n0 = state.dynamic_dimension

    no_contact = ActivePatchSet.empty()
    one_patch = ActivePatchSet(active_patch_ids=(2,), active_load_basis_indices=(0,), alpha=np.array([1.0]))
    multi_patch = ActivePatchSet(
        active_patch_ids=tuple(range(10)),
        active_load_basis_indices=tuple(range(10)),
        alpha=np.arange(10, dtype=float),
    )
    moved_patch = ActivePatchSet(active_patch_ids=(7,), active_load_basis_indices=(0,), alpha=np.array([3.0]))

    dim_y_no_contact = state.as_vector().size + no_contact.dynamic_dimension_contribution
    dim_y_contact = state.as_vector().size + one_patch.dynamic_dimension_contribution
    dim_y_multi_contact = state.as_vector().size + multi_patch.dynamic_dimension_contribution
    dim_y_moved_contact = state.as_vector().size + moved_patch.dynamic_dimension_contribution

    assert dim_y_no_contact == dim_y_contact == dim_y_multi_contact == dim_y_moved_contact == n0
    assert no_contact.alpha_dimension == 0
    assert one_patch.alpha_dimension == 1
    assert multi_patch.alpha_dimension == 10
    assert moved_patch.alpha_dimension == 1
    assert not hasattr(state, "alpha")
    assert not hasattr(state, "q_patch")


def test_ilc_alpha_only_affects_residual_recovery_layer() -> None:
    state = ReducedState(r=np.zeros(3), theta=np.zeros(3), eta_k=np.zeros(4))
    n0 = state.dynamic_dimension
    blocks = {
        3: PatchResidualBlock(patch_id=3, Phi_B_j=np.eye(6, 1)),
        8: PatchResidualBlock(patch_id=8, Phi_B_j=np.roll(np.eye(6, 1), shift=2, axis=0)),
    }

    active_a = ActivePatchSet(active_patch_ids=(3,), active_load_basis_indices=(0,), alpha=np.array([2.0]))
    active_b = ActivePatchSet(active_patch_ids=(8,), active_load_basis_indices=(0,), alpha=np.array([5.0]))

    residual_a = recover_residual_deformation(blocks, active_a)
    residual_b = recover_residual_deformation(blocks, active_b)

    assert state.dynamic_dimension == n0
    assert state.as_vector().size == n0
    np.testing.assert_allclose(residual_a, np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(residual_b, np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]))


def test_contact_force_projection_returns_external_ilc_state() -> None:
    B0 = np.zeros((6, 1), dtype=float)
    B0[2, 0] = 1.0
    B1 = np.zeros((6, 1), dtype=float)
    B1[5, 0] = 1.0
    force = np.zeros(6, dtype=float)
    force[[2, 5]] = [4.0, 7.0]

    active = project_contact_force_to_ilc(
        force,
        [
            PatchLoadBasis(patch_id=0, basis_type="normal", B_j=B0),
            PatchLoadBasis(patch_id=1, basis_type="normal", B_j=B1),
        ],
        regularization=0.0,
    )

    assert active.active_patch_ids == (0, 1)
    assert active.dynamic_dimension_contribution == 0
    np.testing.assert_allclose(active.alpha, [4.0, 7.0])
