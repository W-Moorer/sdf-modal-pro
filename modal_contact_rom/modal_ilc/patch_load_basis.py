from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchLoadBasis:
    """Patch-level interface load coordinate basis.

    ``B_j`` maps the patch ILC amplitudes for one patch into the full force
    vector. It is part of the contact/projection layer, not the dynamic state.
    """

    patch_id: int
    basis_type: str
    B_j: np.ndarray
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.B_j, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("B_j must be a two-dimensional load basis matrix")
        if matrix.shape[0] == 0:
            raise ValueError("B_j must have at least one force DOF")
        if not self.basis_type:
            raise ValueError("basis_type must be non-empty")
        if self.labels is not None and len(self.labels) != matrix.shape[1]:
            raise ValueError("labels length must match the number of load basis columns")
        object.__setattr__(self, "patch_id", int(self.patch_id))
        object.__setattr__(self, "B_j", matrix)

    @property
    def n_dofs(self) -> int:
        return int(self.B_j.shape[0])

    @property
    def load_dimension(self) -> int:
        return int(self.B_j.shape[1])

    @property
    def matrix(self) -> np.ndarray:
        return self.B_j


def assemble_load_basis(load_bases: list[PatchLoadBasis] | tuple[PatchLoadBasis, ...]) -> np.ndarray:
    """Column-stack active patch load bases into ``B_A``."""

    if not load_bases:
        return np.zeros((0, 0), dtype=float)

    n_dofs = load_bases[0].n_dofs
    for basis in load_bases:
        if basis.n_dofs != n_dofs:
            raise ValueError("all patch load bases must have the same number of force DOFs")

    return np.column_stack([basis.B_j for basis in load_bases])
