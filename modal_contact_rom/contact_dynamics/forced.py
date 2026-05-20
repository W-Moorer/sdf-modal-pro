from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from modal_contact_rom.fem_io.generate import FEMSystem


@dataclass(frozen=True)
class ForcedLinearStep:
    time: float
    kinetic_energy: float
    elastic_energy: float
    external_work: float
    energy_balance_error: float


@dataclass(frozen=True)
class ForcedLinearResult:
    steps: list[ForcedLinearStep]
    final_displacement: np.ndarray
    final_velocity: np.ndarray
    displacement_history: np.ndarray
    velocity_history: np.ndarray


def run_full_forced_response(
    fem: FEMSystem,
    force: np.ndarray,
    dt: float,
    steps: int,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
) -> ForcedLinearResult:
    """Run full-order linear dynamics on free DOFs under a constant load."""

    free = fem.free_dofs
    K = fem.K[free, :][:, free].toarray()
    M = fem.M[free, :][:, free].toarray()
    C = rayleigh_alpha * M + rayleigh_beta * K
    q = np.zeros(free.size, dtype=float)
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    full_u = np.zeros(fem.mesh.n_dofs, dtype=float)
    full_v = np.zeros_like(full_u)
    load = force[free]
    history: list[ForcedLinearStep] = []
    displacement_history: list[np.ndarray] = []
    velocity_history: list[np.ndarray] = []
    work = 0.0

    for step_id in range(steps):
        previous_u = full_u.copy()
        q, qd, qdd = _newmark_step(M, K, C, q, qd, qdd, load, dt)
        full_u = np.zeros(fem.mesh.n_dofs, dtype=float)
        full_v = np.zeros_like(full_u)
        full_u[free] = q
        full_v[free] = qd
        displacement_history.append(full_u.copy())
        velocity_history.append(full_v.copy())
        work += float(force @ (full_u - previous_u))
        kinetic = 0.5 * float(qd @ (M @ qd))
        elastic = 0.5 * float(q @ (K @ q))
        history.append(
            ForcedLinearStep(
                time=float((step_id + 1) * dt),
                kinetic_energy=kinetic,
                elastic_energy=elastic,
                external_work=work,
                energy_balance_error=kinetic + elastic - work,
            )
        )

    return ForcedLinearResult(history, full_u, full_v, np.asarray(displacement_history), np.asarray(velocity_history))


def run_reduced_forced_response(
    fem: FEMSystem,
    basis: np.ndarray,
    force: np.ndarray,
    dt: float,
    steps: int,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
) -> ForcedLinearResult:
    """Run Galerkin reduced linear dynamics under a constant load."""

    basis = np.asarray(basis, dtype=float)
    M = basis.T @ (fem.M @ basis)
    K = basis.T @ (fem.K @ basis)
    C = rayleigh_alpha * M + rayleigh_beta * K
    load = basis.T @ force
    q = np.zeros(basis.shape[1], dtype=float)
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    full_u = np.zeros(fem.mesh.n_dofs, dtype=float)
    full_v = np.zeros_like(full_u)
    history: list[ForcedLinearStep] = []
    displacement_history: list[np.ndarray] = []
    velocity_history: list[np.ndarray] = []
    work = 0.0

    for step_id in range(steps):
        previous_u = full_u.copy()
        q, qd, qdd = _newmark_step(M, K, C, q, qd, qdd, load, dt)
        full_u = basis @ q
        full_v = basis @ qd
        displacement_history.append(full_u.copy())
        velocity_history.append(full_v.copy())
        work += float(force @ (full_u - previous_u))
        kinetic = 0.5 * float(qd @ (M @ qd))
        elastic = 0.5 * float(q @ (K @ q))
        history.append(
            ForcedLinearStep(
                time=float((step_id + 1) * dt),
                kinetic_energy=kinetic,
                elastic_energy=elastic,
                external_work=work,
                energy_balance_error=kinetic + elastic - work,
            )
        )

    return ForcedLinearResult(history, full_u, full_v, np.asarray(displacement_history), np.asarray(velocity_history))


def _newmark_step(
    M: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    force: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = 0.25
    gamma = 0.5
    q_predict = q + dt * qd + dt * dt * (0.5 - beta) * qdd
    qd_predict = qd + dt * (1.0 - gamma) * qdd
    lhs = M + gamma * dt * C + beta * dt * dt * K
    rhs = force - C @ qd_predict - K @ q_predict
    qdd_next = la.solve(0.5 * (lhs + lhs.T), rhs, assume_a="sym")
    q_next = q_predict + beta * dt * dt * qdd_next
    qd_next = qd_predict + gamma * dt * qdd_next
    return q_next, qd_next, qdd_next
