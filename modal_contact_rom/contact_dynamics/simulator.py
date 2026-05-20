from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from modal_contact_rom.contact_dynamics.rigid import PrescribedRigidSphere
from modal_contact_rom.fem_io.generate import FEMSystem
from modal_contact_rom.reduced_dynamics.basis import mass_orthonormalize
from modal_contact_rom.surface_patch.extract import SurfaceMesh, SurfacePatch


@dataclass(frozen=True)
class AdaptiveContactStep:
    time: float
    min_gap: float
    max_penetration: float
    normal_force: float
    active_patch_ids: tuple[int, ...]
    kinetic_energy: float
    elastic_energy: float
    contact_work: float
    energy_balance_error: float


@dataclass(frozen=True)
class AdaptiveSimulationResult:
    steps: list[AdaptiveContactStep]
    final_displacement: np.ndarray
    final_velocity: np.ndarray
    displacement_history: np.ndarray | None = None
    velocity_history: np.ndarray | None = None

    @property
    def active_patch_history(self) -> list[tuple[int, ...]]:
        return [step.active_patch_ids for step in self.steps]


class AdaptiveModalContactSimulator:
    """Reduced flexible body with online patch-mode activation."""

    def __init__(
        self,
        fem: FEMSystem,
        surface: SurfaceMesh,
        patches: list[SurfacePatch],
        base_basis: np.ndarray,
        patch_modes: dict[int, np.ndarray],
        rigid_sphere: PrescribedRigidSphere,
        penalty: float,
        contact_damping: float = 0.0,
        rayleigh_alpha: float = 0.0,
        rayleigh_beta: float = 0.0,
        activation_count: int = 1,
        activation_radius: float | None = None,
    ) -> None:
        self.fem = fem
        self.surface = surface
        self.patches = patches
        self.base_basis = np.asarray(base_basis, dtype=float)
        self.patch_modes = patch_modes
        self.rigid_sphere = rigid_sphere
        self.penalty = float(penalty)
        self.contact_damping = float(contact_damping)
        self.rayleigh_alpha = float(rayleigh_alpha)
        self.rayleigh_beta = float(rayleigh_beta)
        self.activation_count = int(activation_count)
        self.activation_radius = activation_radius
        if self.activation_count < 1:
            raise ValueError("activation_count must be positive")

    def run(self, dt: float, steps: int) -> AdaptiveSimulationResult:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if steps < 1:
            raise ValueError("steps must be positive")

        u = np.zeros(self.fem.mesh.n_dofs, dtype=float)
        v = np.zeros_like(u)
        active_patch_ids: tuple[int, ...] = ()
        basis = self._basis_for(active_patch_ids)
        q, qd = self._project_state(basis, u, v)
        qdd = np.zeros_like(q)
        acceleration = np.zeros_like(u)
        history: list[AdaptiveContactStep] = []
        displacement_history: list[np.ndarray] = []
        velocity_history: list[np.ndarray] = []
        contact_work = 0.0

        for step_id in range(steps):
            time = step_id * dt
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
            active_patch_ids = self._activate(contact.contact_points)
            acceleration = basis @ qdd
            basis = self._basis_for(active_patch_ids)
            q, qd = self._project_state(basis, u, v)
            qdd = self._project_vector(basis, acceleration)

            q, qd, qdd = self._newmark_step(basis, q, qd, qdd, contact.vector, dt)
            u = basis @ q
            v = basis @ qd
            displacement_history.append(u.copy())
            velocity_history.append(v.copy())
            contact_work += float(contact.vector @ (u - previous_u))
            kinetic = 0.5 * float(qd @ ((basis.T @ (self.fem.M @ basis)) @ qd))
            elastic = 0.5 * float(q @ ((basis.T @ (self.fem.K @ basis)) @ q))
            mechanical = kinetic + elastic
            history.append(
                AdaptiveContactStep(
                    time=float(time + dt),
                    min_gap=float(self.rigid_sphere.evaluate(self.fem.mesh, self.surface, u, time + dt).min_gap),
                    max_penetration=contact.max_penetration,
                    normal_force=contact.normal_force,
                    active_patch_ids=active_patch_ids,
                    kinetic_energy=kinetic,
                    elastic_energy=elastic,
                    contact_work=contact_work,
                    energy_balance_error=mechanical - contact_work,
                )
            )

        return AdaptiveSimulationResult(
            history,
            u,
            v,
            np.asarray(displacement_history),
            np.asarray(velocity_history),
        )

    def _basis_for(self, active_patch_ids: tuple[int, ...]) -> np.ndarray:
        columns = [self.base_basis]
        for patch_id in active_patch_ids:
            mode = self.patch_modes.get(patch_id)
            if mode is not None:
                columns.append(mode.reshape(-1, 1))
        raw = np.column_stack(columns)
        basis = mass_orthonormalize(self.fem.M, raw)
        if basis.shape[1] == 0:
            raise RuntimeError("active reduced basis is empty")
        return basis

    def _activate(self, contact_points: np.ndarray) -> tuple[int, ...]:
        if contact_points.size == 0:
            return ()
        distances: list[tuple[float, int]] = []
        for patch in self.patches:
            point_distances = np.linalg.norm(contact_points - patch.centroid, axis=1)
            distance = float(np.min(point_distances))
            if self.activation_radius is None or distance <= self.activation_radius:
                distances.append((distance, patch.patch_id))
        distances.sort()
        return tuple(patch_id for _, patch_id in distances[: self.activation_count])

    def _project_state(self, basis: np.ndarray, displacement: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Mr = basis.T @ (self.fem.M @ basis)
        q = la.solve(Mr, basis.T @ (self.fem.M @ displacement), assume_a="sym")
        qd = la.solve(Mr, basis.T @ (self.fem.M @ velocity), assume_a="sym")
        return q, qd

    def _project_vector(self, basis: np.ndarray, vector: np.ndarray) -> np.ndarray:
        Mr = basis.T @ (self.fem.M @ basis)
        return la.solve(Mr, basis.T @ (self.fem.M @ vector), assume_a="sym")

    def _newmark_step(
        self,
        basis: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        force: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        beta = 0.25
        gamma = 0.5
        Mr = basis.T @ (self.fem.M @ basis)
        Kr = basis.T @ (self.fem.K @ basis)
        Cr = self.rayleigh_alpha * Mr + self.rayleigh_beta * Kr
        Q = basis.T @ force

        q_predict = q + dt * qd + dt * dt * (0.5 - beta) * qdd
        qd_predict = qd + dt * (1.0 - gamma) * qdd
        lhs = Mr + gamma * dt * Cr + beta * dt * dt * Kr
        rhs = Q - Cr @ qd_predict - Kr @ q_predict
        qdd_next = la.solve(0.5 * (lhs + lhs.T), rhs, assume_a="sym")
        q_next = q_predict + beta * dt * dt * qdd_next
        qd_next = qd_predict + gamma * dt * qdd_next
        return q_next, qd_next, qdd_next


def patch_mode_dict(patch_ids: np.ndarray, modes: np.ndarray) -> dict[int, np.ndarray]:
    return {int(patch_id): modes[:, index] for index, patch_id in enumerate(patch_ids)}
