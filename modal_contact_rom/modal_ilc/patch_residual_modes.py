from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchResidualBlock:
    """Residual deformation block associated with one active contact patch."""

    patch_id: int
    Phi_B_j: np.ndarray

    def __post_init__(self) -> None:
        modes = np.asarray(self.Phi_B_j, dtype=float)
        if modes.ndim != 2:
            raise ValueError("Phi_B_j must be a two-dimensional residual mode matrix")
        if modes.shape[0] == 0:
            raise ValueError("Phi_B_j must have at least one displacement DOF")
        object.__setattr__(self, "patch_id", int(self.patch_id))
        object.__setattr__(self, "Phi_B_j", modes)

    @property
    def n_dofs(self) -> int:
        return int(self.Phi_B_j.shape[0])

    @property
    def alpha_dimension(self) -> int:
        return int(self.Phi_B_j.shape[1])

    @property
    def modes(self) -> np.ndarray:
        return self.Phi_B_j

    def deformation(self, alpha: np.ndarray) -> np.ndarray:
        alpha_vector = np.asarray(alpha, dtype=float)
        if alpha_vector.ndim != 1:
            raise ValueError("alpha must be a one-dimensional vector")
        if alpha_vector.size != self.alpha_dimension:
            raise ValueError("alpha length must match the residual block dimension")
        return self.Phi_B_j @ alpha_vector
