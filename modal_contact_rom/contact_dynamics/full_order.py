from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.linalg as la

from modal_contact_rom.contact_dynamics.rigid import PrescribedRigidPlane, PrescribedRigidSphere
from modal_contact_rom.fem_io.generate import FEMSystem
from modal_contact_rom.surface_patch.extract import SurfaceMesh


@dataclass(frozen=True)
class FullFEMContactStep:
    time: float
    min_gap: float
    max_penetration: float
    normal_force: float
    kinetic_energy: float
    elastic_energy: float
    contact_work: float
    external_work: float
    energy_balance_error: float


@dataclass(frozen=True)
class FullFEMSimulationResult:
    steps: list[FullFEMContactStep]
    final_displacement: np.ndarray
    final_velocity: np.ndarray
    displacement_history: np.ndarray
    velocity_history: np.ndarray


class FullFEMContactSimulator:
    """Full-order free-DOF contact dynamics reference model."""

    def __init__(
        self,
        fem: FEMSystem,
        surface: SurfaceMesh,
        rigid_sphere: PrescribedRigidSphere | PrescribedRigidPlane,
        penalty: float,
        contact_damping: float = 0.0,
        rayleigh_alpha: float = 0.0,
        rayleigh_beta: float = 0.0,
        external_force: np.ndarray | Callable[[float], np.ndarray] | None = None,
    ) -> None:
        self.fem = fem
        self.surface = surface
        self.rigid_sphere = rigid_sphere
        self.penalty = float(penalty)
        self.contact_damping = float(contact_damping)
        self.rayleigh_alpha = float(rayleigh_alpha)
        self.rayleigh_beta = float(rayleigh_beta)
        self.external_force = external_force
        if external_force is not None and not callable(external_force):
            external_force_array = np.asarray(external_force, dtype=float)
            if external_force_array.shape != (fem.mesh.n_dofs,):
                raise ValueError(f"external_force must have shape ({fem.mesh.n_dofs},)")
            self.external_force = external_force_array

    def run(self, dt: float, steps: int) -> FullFEMSimulationResult:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if steps < 1:
            raise ValueError("steps must be positive")

        free = self.fem.free_dofs
        Kff = self.fem.K[free, :][:, free].toarray()
        Mff = self.fem.M[free, :][:, free].toarray()
        Cff = self.rayleigh_alpha * Mff + self.rayleigh_beta * Kff

        q = np.zeros(free.size, dtype=float)
        qd = np.zeros_like(q)
        qdd = np.zeros_like(q)
        u = np.zeros(self.fem.mesh.n_dofs, dtype=float)
        v = np.zeros_like(u)
        history: list[FullFEMContactStep] = []
        displacement_history: list[np.ndarray] = []
        velocity_history: list[np.ndarray] = []
        contact_work = 0.0
        external_work = 0.0
        previous_external_force = self._external_force(0.0)

        for step_id in range(steps):
            time = step_id * dt
            load_time = time + dt
            previous_u = u.copy()
            contact = self.rigid_sphere.contact_force(
                self.fem.mesh,
                self.surface,
                u,
                v,
                time,
                penalty=self.penalty,
                damping=self.contact_damping,
            )
            current_external_force = self._external_force(load_time)
            total_force = contact.vector + current_external_force
            q, qd, qdd = _newmark_step(Mff, Kff, Cff, q, qd, qdd, total_force[free], dt)
            u = np.zeros(self.fem.mesh.n_dofs, dtype=float)
            v = np.zeros_like(u)
            u[free] = q
            v[free] = qd
            displacement_history.append(u.copy())
            velocity_history.append(v.copy())
            displacement_increment = u - previous_u
            contact_work += float(contact.vector @ displacement_increment)
            external_work += float(0.5 * (previous_external_force + current_external_force) @ displacement_increment)
            previous_external_force = current_external_force
            kinetic = 0.5 * float(qd @ (Mff @ qd))
            elastic = 0.5 * float(q @ (Kff @ q))
            mechanical = kinetic + elastic
            history.append(
                FullFEMContactStep(
                    time=float(time + dt),
                    min_gap=float(self.rigid_sphere.evaluate(self.fem.mesh, self.surface, u, time + dt).min_gap),
                    max_penetration=contact.max_penetration,
                    normal_force=contact.normal_force,
                    kinetic_energy=kinetic,
                    elastic_energy=elastic,
                    contact_work=contact_work,
                    external_work=external_work,
                    energy_balance_error=mechanical - contact_work - external_work,
                )
            )

        return FullFEMSimulationResult(
            history,
            u,
            v,
            np.asarray(displacement_history),
            np.asarray(velocity_history),
        )

    def _external_force(self, time: float) -> np.ndarray:
        if self.external_force is None:
            return np.zeros(self.fem.mesh.n_dofs, dtype=float)
        if callable(self.external_force):
            force = np.asarray(self.external_force(time), dtype=float)
        else:
            force = np.asarray(self.external_force, dtype=float)
        if force.shape != (self.fem.mesh.n_dofs,):
            raise ValueError(f"external_force must have shape ({self.fem.mesh.n_dofs},)")
        return force


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
