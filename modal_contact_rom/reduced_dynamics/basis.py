from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


def mass_orthonormalize(
    M: sp.spmatrix,
    vectors: np.ndarray,
    rtol: float = 1.0e-10,
) -> np.ndarray:
    """Return an M-orthonormal basis spanning the input vectors."""

    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if vectors.shape[1] == 0:
        return np.zeros((vectors.shape[0], 0), dtype=float)

    gram = vectors.T @ (M @ vectors)
    gram = 0.5 * (gram + gram.T)
    values, transform = la.eigh(gram)
    if values.size == 0:
        return np.zeros((vectors.shape[0], 0), dtype=float)

    threshold = max(float(np.max(values)) * rtol, rtol)
    keep = values > threshold
    if not np.any(keep):
        return np.zeros((vectors.shape[0], 0), dtype=float)

    return vectors @ transform[:, keep] @ np.diag(1.0 / np.sqrt(values[keep]))


def reduced_static_response(K: sp.spmatrix, basis: np.ndarray, load: np.ndarray) -> np.ndarray:
    """Solve the Galerkin reduced static system for one load vector."""

    if basis.shape[1] == 0:
        return np.zeros(K.shape[0], dtype=float)
    Kr = basis.T @ (K @ basis)
    fr = basis.T @ load
    qr = la.solve(0.5 * (Kr + Kr.T), fr, assume_a="sym")
    return basis @ qr

