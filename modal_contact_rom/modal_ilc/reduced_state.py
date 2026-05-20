from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReducedState:
    """Main dynamic state for the patch-ILC formulation.

    Phase 1 enforces that patch ILCs are not dynamic unknowns.  The solver state
    is limited to rigid translation, rigid rotation parameters, retained modal
    coordinates, and optional constraint multipliers.
    """

    r: np.ndarray
    theta: np.ndarray
    eta_k: np.ndarray
    lambda_: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", _as_vector(self.r, "r"))
        object.__setattr__(self, "theta", _as_vector(self.theta, "theta"))
        object.__setattr__(self, "eta_k", _as_vector(self.eta_k, "eta_k"))
        if self.lambda_ is not None:
            object.__setattr__(self, "lambda_", _as_vector(self.lambda_, "lambda_"))

    @property
    def dynamic_dimension(self) -> int:
        multipliers = 0 if self.lambda_ is None else int(self.lambda_.size)
        return int(self.r.size + self.theta.size + self.eta_k.size + multipliers)

    @property
    def translation(self) -> np.ndarray:
        return self.r

    @property
    def rotation(self) -> np.ndarray:
        return self.theta

    @property
    def lagrange_multipliers(self) -> np.ndarray | None:
        return self.lambda_

    def as_vector(self) -> np.ndarray:
        parts = [
            self.r,
            self.theta,
            np.asarray(self.eta_k, dtype=float).reshape(-1),
        ]
        if self.lambda_ is not None:
            parts.append(self.lambda_)
        return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return array
