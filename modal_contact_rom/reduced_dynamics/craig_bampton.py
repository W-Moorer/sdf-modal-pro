from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class CraigBamptonResult:
    interface_dofs: np.ndarray
    interior_dofs: np.ndarray
    constraint_modes: np.ndarray
    fixed_interface_modes: np.ndarray
    basis: np.ndarray
    eigenvalues: np.ndarray
    frequencies_hz: np.ndarray


def compute_craig_bampton_basis(
    K: sp.spmatrix,
    M: sp.spmatrix,
    free_dofs: np.ndarray,
    interface_dofs: np.ndarray,
    num_fixed_interface_modes: int,
) -> CraigBamptonResult:
    """Build a Craig-Bampton basis for one flexible component.

    The returned basis contains static constraint modes for all interface DOFs,
    followed by fixed-interface vibration modes on the remaining interior DOFs.
    """

    free_dofs = np.asarray(free_dofs, dtype=np.int64)
    interface_dofs = np.asarray(interface_dofs, dtype=np.int64)
    interface_dofs = np.intersect1d(interface_dofs, free_dofs, assume_unique=False)
    if interface_dofs.size == 0:
        raise ValueError("interface_dofs must contain at least one free dof")
    if num_fixed_interface_modes < 0:
        raise ValueError("num_fixed_interface_modes must be nonnegative")

    interior_dofs = np.setdiff1d(free_dofs, interface_dofs, assume_unique=False)
    if interior_dofs.size == 0:
        raise ValueError("Craig-Bampton reduction requires interior dofs")

    Kii = K[interior_dofs, :][:, interior_dofs].tocsc()
    Kib = K[interior_dofs, :][:, interface_dofs].tocsc()
    Mii = M[interior_dofs, :][:, interior_dofs].tocsc()
    solve_kii = spla.factorized(Kii)

    constraint_modes = np.zeros((K.shape[0], interface_dofs.size), dtype=float)
    constraint_modes[interface_dofs, :] = np.eye(interface_dofs.size)
    constraint_modes[interior_dofs, :] = -solve_kii(Kib.toarray())

    mode_count = min(num_fixed_interface_modes, max(interior_dofs.size - 1, 0))
    eigenvalues = np.empty(0, dtype=float)
    frequencies = np.empty(0, dtype=float)
    fixed_interface_modes = np.zeros((K.shape[0], 0), dtype=float)
    if mode_count > 0:
        try:
            values, vectors = spla.eigsh(Kii, k=mode_count, M=Mii, which="SM")
        except Exception:
            values, vectors = la.eigh(Kii.toarray(), Mii.toarray(), subset_by_index=(0, mode_count - 1))
        order = np.argsort(values)
        eigenvalues = np.asarray(values[order], dtype=float)
        vectors = np.asarray(vectors[:, order], dtype=float)
        fixed_interface_modes = np.zeros((K.shape[0], mode_count), dtype=float)
        fixed_interface_modes[interior_dofs, :] = vectors
        frequencies = np.sqrt(np.maximum(eigenvalues, 0.0)) / (2.0 * np.pi)

    basis = np.column_stack((constraint_modes, fixed_interface_modes))
    return CraigBamptonResult(
        interface_dofs=interface_dofs,
        interior_dofs=interior_dofs,
        constraint_modes=constraint_modes,
        fixed_interface_modes=fixed_interface_modes,
        basis=basis,
        eigenvalues=eigenvalues,
        frequencies_hz=frequencies,
    )
