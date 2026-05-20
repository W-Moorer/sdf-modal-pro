from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class ModalResult:
    eigenvalues: np.ndarray
    frequencies_hz: np.ndarray
    modes: np.ndarray


def compute_low_modes(
    K: sp.spmatrix,
    M: sp.spmatrix,
    free_dofs: np.ndarray,
    num_modes: int,
) -> ModalResult:
    """Solve K phi = lambda M phi on free dofs and expand to full dofs."""

    if num_modes < 1:
        raise ValueError("num_modes must be positive")

    free_dofs = np.asarray(free_dofs, dtype=np.int64)
    n_free = int(free_dofs.size)
    if n_free < 2:
        raise ValueError("at least two free dofs are required")

    k = min(num_modes, n_free - 1)
    Kff = K[free_dofs, :][:, free_dofs].tocsc()
    Mff = M[free_dofs, :][:, free_dofs].tocsc()

    try:
        values, vectors = spla.eigsh(Kff, k=k, M=Mff, which="SM")
    except Exception:
        values, vectors = la.eigh(Kff.toarray(), Mff.toarray(), subset_by_index=(0, k - 1))

    order = np.argsort(values)
    values = np.asarray(values[order], dtype=float)
    vectors = np.asarray(vectors[:, order], dtype=float)

    full = np.zeros((K.shape[0], k), dtype=float)
    full[free_dofs, :] = vectors
    full = _mass_normalize(full, M)
    frequencies = np.sqrt(np.maximum(values, 0.0)) / (2.0 * np.pi)
    return ModalResult(values, frequencies, full)


def _mass_normalize(vectors: np.ndarray, M: sp.spmatrix) -> np.ndarray:
    vectors = np.array(vectors, dtype=float, copy=True)
    for i in range(vectors.shape[1]):
        norm = float(np.sqrt(vectors[:, i].T @ (M @ vectors[:, i])))
        if norm > 0.0:
            vectors[:, i] /= norm
    return vectors

