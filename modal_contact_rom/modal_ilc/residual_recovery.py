from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from modal_contact_rom.modal_ilc.ilc_projection import ActivePatchSet
from modal_contact_rom.modal_ilc.patch_residual_modes import PatchResidualBlock


def recover_residual_deformation(
    residual_blocks: Mapping[int, PatchResidualBlock] | Sequence[PatchResidualBlock],
    active_set: ActivePatchSet,
    n_dofs: int | None = None,
) -> np.ndarray:
    """Recover ``Phi_B,A alpha_A`` without changing the dynamic state vector."""

    blocks = _block_mapping(residual_blocks)
    if not blocks:
        if n_dofs is None:
            return np.zeros(0, dtype=float)
        return np.zeros(int(n_dofs), dtype=float)

    block_dofs = _infer_n_dofs(blocks)
    if n_dofs is None:
        n_dofs = block_dofs
    elif int(n_dofs) != block_dofs:
        raise ValueError("n_dofs must match the residual block displacement DOFs")

    residual = np.zeros(int(n_dofs), dtype=float)
    cursor = 0
    for patch_id in active_set.active_patch_ids:
        if patch_id not in blocks:
            raise KeyError(f"missing residual block for active patch {patch_id}")
        block = blocks[patch_id]
        next_cursor = cursor + block.alpha_dimension
        alpha_j = active_set.alpha[cursor:next_cursor]
        if alpha_j.size != block.alpha_dimension:
            raise ValueError("active_set alpha is too short for the active residual blocks")
        residual += block.deformation(alpha_j)
        cursor = next_cursor

    if cursor != active_set.alpha.size:
        raise ValueError("active_set alpha has unused entries")
    return residual


def recover_total_deformation(
    retained_modes: np.ndarray,
    eta_k: np.ndarray,
    residual_blocks: Mapping[int, PatchResidualBlock] | Sequence[PatchResidualBlock],
    active_set: ActivePatchSet,
) -> np.ndarray:
    """Recover ``Psi_k eta_k + Phi_B,A alpha_A``."""

    Psi = np.asarray(retained_modes, dtype=float)
    eta = np.asarray(eta_k, dtype=float)
    if Psi.ndim != 2:
        raise ValueError("retained_modes must be a two-dimensional matrix")
    if eta.ndim != 1:
        raise ValueError("eta_k must be a one-dimensional vector")
    if Psi.shape[1] != eta.size:
        raise ValueError("eta_k length must match retained_modes columns")

    modal = Psi @ eta
    residual = recover_residual_deformation(residual_blocks, active_set, n_dofs=Psi.shape[0])
    return modal + residual


def _block_mapping(
    residual_blocks: Mapping[int, PatchResidualBlock] | Sequence[PatchResidualBlock],
) -> dict[int, PatchResidualBlock]:
    if isinstance(residual_blocks, Mapping):
        return {int(patch_id): block for patch_id, block in residual_blocks.items()}
    return {block.patch_id: block for block in residual_blocks}


def _infer_n_dofs(blocks: Mapping[int, PatchResidualBlock]) -> int:
    n_dofs = next(iter(blocks.values())).n_dofs
    for block in blocks.values():
        if block.n_dofs != n_dofs:
            raise ValueError("all residual blocks must have the same displacement DOFs")
    return n_dofs
