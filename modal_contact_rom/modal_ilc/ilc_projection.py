from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from modal_contact_rom.modal_ilc.patch_load_basis import PatchLoadBasis, assemble_load_basis


@dataclass(frozen=True)
class ActivePatchSet:
    """Active contact patches and their interface load coordinates.

    ``alpha`` belongs to the contact/residual layer. It is intentionally kept
    outside ``ReducedState`` so activation changes do not resize the main DAE.
    """

    active_patch_ids: tuple[int, ...]
    active_load_basis_indices: tuple[int, ...]
    alpha: np.ndarray

    def __post_init__(self) -> None:
        alpha = np.asarray(self.alpha, dtype=float)
        if alpha.ndim != 1:
            raise ValueError("alpha must be a one-dimensional vector")
        if self.active_load_basis_indices and len(self.active_load_basis_indices) != alpha.size:
            raise ValueError("active_load_basis_indices length must match alpha length")
        object.__setattr__(self, "active_patch_ids", tuple(int(patch_id) for patch_id in self.active_patch_ids))
        object.__setattr__(
            self,
            "active_load_basis_indices",
            tuple(int(index) for index in self.active_load_basis_indices),
        )
        object.__setattr__(self, "alpha", alpha)

    @classmethod
    def empty(cls) -> ActivePatchSet:
        return cls(active_patch_ids=(), active_load_basis_indices=(), alpha=np.zeros(0, dtype=float))

    @property
    def alpha_dimension(self) -> int:
        return int(self.alpha.size)

    @property
    def dynamic_dimension_contribution(self) -> int:
        return 0

    @property
    def is_empty(self) -> bool:
        return self.alpha.size == 0 and len(self.active_patch_ids) == 0


def project_contact_force_to_ilc(
    contact_force: np.ndarray,
    load_bases: Sequence[PatchLoadBasis],
    weights: np.ndarray | None = None,
    regularization: float = 1.0e-8,
) -> ActivePatchSet:
    """Project the current contact force onto active patch ILC coordinates."""

    if not load_bases:
        return ActivePatchSet.empty()

    force = np.asarray(contact_force, dtype=float)
    if force.ndim != 1:
        raise ValueError("contact_force must be a one-dimensional vector")

    B = assemble_load_basis(tuple(load_bases))
    if B.shape[0] != force.size:
        raise ValueError("contact_force length must match the load basis force DOFs")

    normal_matrix, rhs = _weighted_normal_equations(B, force, weights)
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if regularization > 0.0:
        normal_matrix = normal_matrix + float(regularization) * np.eye(normal_matrix.shape[0])

    alpha = np.linalg.solve(normal_matrix, rhs)
    active_patch_ids = tuple(dict.fromkeys(basis.patch_id for basis in load_bases))
    active_load_basis_indices = tuple(range(alpha.size))
    return ActivePatchSet(active_patch_ids, active_load_basis_indices, alpha)


def _weighted_normal_equations(
    B: np.ndarray,
    force: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        return B.T @ B, B.T @ force

    W = np.asarray(weights, dtype=float)
    if W.ndim == 1:
        if W.size != force.size:
            raise ValueError("weights vector length must match contact_force length")
        if np.any(W < 0.0):
            raise ValueError("weights must be non-negative")
        return B.T @ (B * W[:, None]), B.T @ (force * W)
    if W.ndim == 2:
        if W.shape != (force.size, force.size):
            raise ValueError("weights matrix shape must match contact_force length")
        return B.T @ (W @ B), B.T @ (W @ force)
    raise ValueError("weights must be either a vector or a square matrix")
