from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from modal_contact_rom.contact_modes.loads import PatchLoad


@dataclass(frozen=True)
class ContactModeResult:
    patch_ids: np.ndarray
    modes: np.ndarray


def solve_patch_contact_modes(
    K: sp.spmatrix,
    loads: list[PatchLoad],
    free_dofs: np.ndarray,
    min_rhs_norm: float = 1.0e-12,
) -> ContactModeResult:
    """Solve K psi_j = b_j for each patch load on constrained free dofs."""

    free_dofs = np.asarray(free_dofs, dtype=np.int64)
    Kff = K[free_dofs, :][:, free_dofs].tocsc()
    solve = spla.factorized(Kff)
    modes: list[np.ndarray] = []
    patch_ids: list[int] = []

    for load in loads:
        rhs = load.vector[free_dofs]
        if np.linalg.norm(rhs) <= min_rhs_norm:
            continue
        mode = np.zeros(K.shape[0], dtype=float)
        mode[free_dofs] = solve(rhs)
        modes.append(mode)
        patch_ids.append(load.patch_id)

    if not modes:
        return ContactModeResult(np.empty(0, dtype=np.int64), np.zeros((K.shape[0], 0), dtype=float))
    return ContactModeResult(np.asarray(patch_ids, dtype=np.int64), np.column_stack(modes))

