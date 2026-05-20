from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from modal_contact_rom.contact_dynamics.rigid import PrescribedRigidPlane, PrescribedRigidSphere
from modal_contact_rom.fem_io.generate import FEMSystem
from modal_contact_rom.reduced_dynamics.basis import mass_orthonormalize
from modal_contact_rom.surface_patch.extract import SurfaceMesh, SurfacePatch


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


@dataclass(frozen=True)
class AdaptiveROMContactStep:
    time: float
    min_gap: float
    max_penetration: float
    normal_force: float
    active_patch_ids: tuple[int, ...]
    basis_size: int
    activation_iterations: int
    kinetic_energy: float
    elastic_energy: float
    contact_work: float
    external_work: float
    energy_balance_error: float


@dataclass(frozen=True)
class AdaptiveROMSimulationResult:
    steps: list[AdaptiveROMContactStep]
    final_displacement: np.ndarray
    final_velocity: np.ndarray
    displacement_history: np.ndarray
    velocity_history: np.ndarray

    @property
    def active_patch_history(self) -> list[tuple[int, ...]]:
        return [step.active_patch_ids for step in self.steps]

    @property
    def basis_size_history(self) -> list[int]:
        return [step.basis_size for step in self.steps]


@dataclass(frozen=True)
class _QuadraturePlaneContact:
    vector: np.ndarray
    tangent: sp.csr_matrix
    min_gap: float
    max_penetration: float
    normal_force: float
    energy: float
    contact_node_ids: np.ndarray
    contact_points: np.ndarray


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


class CalculixAlignedFullFEMContactSimulator:
    """Full-order implicit plane contact with CalculiX-style surface quadrature.

    The legacy ``FullFEMContactSimulator`` is intentionally simple: it evaluates
    nodal contact once at the beginning of the step.  This class is the
    validation path used when the external reference is CalculiX native contact:
    contact is integrated on exterior HEX8 faces, contributes a tangent, and is
    solved inside the Newmark/HHT Newton iteration.
    """

    def __init__(
        self,
        fem: FEMSystem,
        rigid_plane: PrescribedRigidPlane,
        contact_stiffness: float,
        rayleigh_alpha: float = 0.0,
        rayleigh_beta: float = 0.0,
        external_force: np.ndarray | Callable[[float], np.ndarray] | None = None,
        hht_alpha: float = 0.0,
        max_iterations: int = 20,
        tolerance: float = 1.0e-11,
    ) -> None:
        if contact_stiffness <= 0.0:
            raise ValueError("contact_stiffness must be positive")
        if not (-1.0 / 3.0 <= hht_alpha <= 0.0):
            raise ValueError("hht_alpha must lie in [-1/3, 0]")
        self.fem = fem
        self.rigid_plane = rigid_plane
        self.contact_stiffness = float(contact_stiffness)
        self.rayleigh_alpha = float(rayleigh_alpha)
        self.rayleigh_beta = float(rayleigh_beta)
        self.hht_alpha = float(hht_alpha)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.external_force = external_force
        if external_force is not None and not callable(external_force):
            external_force_array = np.asarray(external_force, dtype=float)
            if external_force_array.shape != (fem.mesh.n_dofs,):
                raise ValueError(f"external_force must have shape ({fem.mesh.n_dofs},)")
            self.external_force = external_force_array
        self._contact_faces = _plane_aligned_hex_boundary_faces(fem.mesh.nodes, fem.mesh.elements, rigid_plane.normal)
        if self._contact_faces.size == 0:
            raise ValueError("no exterior HEX8 faces align with the contact plane normal")

    def run(self, dt: float, steps: int) -> FullFEMSimulationResult:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if steps < 1:
            raise ValueError("steps must be positive")

        free = self.fem.free_dofs
        Kff = self.fem.K[free, :][:, free].tocsr()
        Mff = self.fem.M[free, :][:, free].tocsr()
        Cff = (self.rayleigh_alpha * Mff + self.rayleigh_beta * Kff).tocsr()
        alpha = self.hht_alpha
        beta = 0.25 * (1.0 - alpha) ** 2
        gamma = 0.5 - alpha

        u = np.zeros(self.fem.mesh.n_dofs, dtype=float)
        v = np.zeros_like(u)
        contact0 = self._contact(u)
        fext0 = self._external_force(0.0)
        q = u[free]
        qd = v[free]
        qdd = np.asarray(spsolve(Mff.tocsc(), fext0[free] + contact0.vector[free] - Kff @ q - Cff @ qd), dtype=float)
        previous_static = Kff @ q + Cff @ qd - fext0[free] - contact0.vector[free]

        history: list[FullFEMContactStep] = []
        displacement_history: list[np.ndarray] = []
        velocity_history: list[np.ndarray] = []
        external_work = 0.0
        contact_work = 0.0
        previous_external_force = fext0

        for step_id in range(steps):
            time = step_id * dt
            load_time = time + dt
            previous_u = u.copy()
            current_external_force = self._external_force(load_time)

            q_predict = q + dt * qd + dt * dt * (0.5 - beta) * qdd
            qd_predict = qd + dt * (1.0 - gamma) * qdd
            c0 = 1.0 / (beta * dt * dt)
            c1 = gamma / (beta * dt)
            q_guess = q_predict.copy()

            for _iteration in range(max(1, self.max_iterations)):
                u_guess = np.zeros(self.fem.mesh.n_dofs, dtype=float)
                u_guess[free] = q_guess
                contact = self._contact(u_guess)
                a_guess = c0 * (q_guess - q_predict)
                v_guess = qd_predict + gamma * dt * a_guess
                static = Kff @ q_guess + Cff @ v_guess - current_external_force[free] - contact.vector[free]
                residual = Mff @ a_guess + (1.0 + alpha) * static - alpha * previous_static
                tangent = (
                    Mff * c0
                    + (Kff + Cff * c1 + contact.tangent[free, :][:, free]) * (1.0 + alpha)
                ).tocsc()
                correction = np.asarray(spsolve(tangent, -residual), dtype=float)
                q_guess += correction
                if np.linalg.norm(correction) <= self.tolerance * max(1.0, np.linalg.norm(q_guess)):
                    break

            q = q_guess
            qdd = c0 * (q - q_predict)
            qd = qd_predict + gamma * dt * qdd
            u = np.zeros(self.fem.mesh.n_dofs, dtype=float)
            v = np.zeros_like(u)
            u[free] = q
            v[free] = qd
            contact = self._contact(u)
            previous_static = Kff @ q + Cff @ qd - current_external_force[free] - contact.vector[free]

            displacement_history.append(u.copy())
            velocity_history.append(v.copy())
            displacement_increment = u - previous_u
            contact_work += float(contact.vector @ displacement_increment)
            external_work += float(0.5 * (previous_external_force + current_external_force) @ displacement_increment)
            previous_external_force = current_external_force
            kinetic = 0.5 * float(qd @ (Mff @ qd))
            elastic = 0.5 * float(q @ (Kff @ q))
            mechanical = kinetic + elastic + contact.energy
            history.append(
                FullFEMContactStep(
                    time=float(load_time),
                    min_gap=contact.min_gap,
                    max_penetration=contact.max_penetration,
                    normal_force=contact.normal_force,
                    kinetic_energy=kinetic,
                    elastic_energy=elastic,
                    contact_work=contact_work,
                    external_work=external_work,
                    energy_balance_error=mechanical - external_work,
                )
            )

        return FullFEMSimulationResult(history, u, v, np.asarray(displacement_history), np.asarray(velocity_history))

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

    def _contact(self, displacement: np.ndarray) -> _QuadraturePlaneContact:
        return _surface_quadrature_plane_contact(
            self.fem.mesh.nodes,
            self._contact_faces,
            displacement,
            point=self.rigid_plane.point,
            normal=self.rigid_plane.normal,
            stiffness=self.contact_stiffness,
        )


class CalculixAlignedROMContactSimulator:
    """Projected ROM using the same contact residual as the aligned full model."""

    def __init__(
        self,
        fem: FEMSystem,
        basis: np.ndarray,
        rigid_plane: PrescribedRigidPlane,
        contact_stiffness: float,
        rayleigh_alpha: float = 0.0,
        rayleigh_beta: float = 0.0,
        external_force: np.ndarray | Callable[[float], np.ndarray] | None = None,
        hht_alpha: float = 0.0,
        max_iterations: int = 20,
        tolerance: float = 1.0e-11,
    ) -> None:
        if contact_stiffness <= 0.0:
            raise ValueError("contact_stiffness must be positive")
        if not (-1.0 / 3.0 <= hht_alpha <= 0.0):
            raise ValueError("hht_alpha must lie in [-1/3, 0]")
        basis_array = np.asarray(basis, dtype=float)
        if basis_array.ndim != 2 or basis_array.shape[0] != fem.mesh.n_dofs:
            raise ValueError(f"basis must have shape ({fem.mesh.n_dofs}, n_modes)")
        if basis_array.shape[1] == 0:
            raise ValueError("basis must contain at least one mode")
        self.fem = fem
        self.basis = basis_array
        self.rigid_plane = rigid_plane
        self.contact_stiffness = float(contact_stiffness)
        self.rayleigh_alpha = float(rayleigh_alpha)
        self.rayleigh_beta = float(rayleigh_beta)
        self.hht_alpha = float(hht_alpha)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.external_force = external_force
        if external_force is not None and not callable(external_force):
            external_force_array = np.asarray(external_force, dtype=float)
            if external_force_array.shape != (fem.mesh.n_dofs,):
                raise ValueError(f"external_force must have shape ({fem.mesh.n_dofs},)")
            self.external_force = external_force_array
        self._contact_faces = _plane_aligned_hex_boundary_faces(fem.mesh.nodes, fem.mesh.elements, rigid_plane.normal)
        if self._contact_faces.size == 0:
            raise ValueError("no exterior HEX8 faces align with the contact plane normal")

    def run(self, dt: float, steps: int) -> FullFEMSimulationResult:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if steps < 1:
            raise ValueError("steps must be positive")

        basis = self.basis
        Kr = np.asarray(basis.T @ (self.fem.K @ basis), dtype=float)
        Mr = np.asarray(basis.T @ (self.fem.M @ basis), dtype=float)
        Cr = self.rayleigh_alpha * Mr + self.rayleigh_beta * Kr
        alpha = self.hht_alpha
        beta = 0.25 * (1.0 - alpha) ** 2
        gamma = 0.5 - alpha

        q = np.zeros(basis.shape[1], dtype=float)
        qd = np.zeros_like(q)
        u = basis @ q
        contact0 = self._contact(u)
        fext0 = self._external_force(0.0)
        qdd = la.solve(Mr, basis.T @ (fext0 + contact0.vector - self.fem.K @ u), assume_a="sym")
        previous_static = Kr @ q + Cr @ qd - basis.T @ (fext0 + contact0.vector)

        history: list[FullFEMContactStep] = []
        displacement_history: list[np.ndarray] = []
        velocity_history: list[np.ndarray] = []
        external_work = 0.0
        contact_work = 0.0
        previous_external_force = fext0

        for step_id in range(steps):
            load_time = (step_id + 1) * dt
            previous_u = basis @ q
            current_external_force = self._external_force(load_time)
            reduced_external = basis.T @ current_external_force

            q_predict = q + dt * qd + dt * dt * (0.5 - beta) * qdd
            qd_predict = qd + dt * (1.0 - gamma) * qdd
            c0 = 1.0 / (beta * dt * dt)
            c1 = gamma / (beta * dt)
            q_guess = q_predict.copy()

            for _iteration in range(max(1, self.max_iterations)):
                u_guess = basis @ q_guess
                contact = self._contact(u_guess)
                a_guess = c0 * (q_guess - q_predict)
                v_guess = qd_predict + gamma * dt * a_guess
                static = Kr @ q_guess + Cr @ v_guess - reduced_external - basis.T @ contact.vector
                residual = Mr @ a_guess + (1.0 + alpha) * static - alpha * previous_static
                contact_tangent = np.asarray(basis.T @ (contact.tangent @ basis), dtype=float)
                tangent = Mr * c0 + (Kr + Cr * c1 + contact_tangent) * (1.0 + alpha)
                correction = la.solve(0.5 * (tangent + tangent.T), -residual, assume_a="sym")
                q_guess += correction
                if np.linalg.norm(correction) <= self.tolerance * max(1.0, np.linalg.norm(q_guess)):
                    break

            q = q_guess
            qdd = c0 * (q - q_predict)
            qd = qd_predict + gamma * dt * qdd
            u = basis @ q
            v = basis @ qd
            contact = self._contact(u)
            previous_static = Kr @ q + Cr @ qd - basis.T @ (current_external_force + contact.vector)

            displacement_history.append(u.copy())
            velocity_history.append(v.copy())
            displacement_increment = u - previous_u
            contact_work += float(contact.vector @ displacement_increment)
            external_work += float(0.5 * (previous_external_force + current_external_force) @ displacement_increment)
            previous_external_force = current_external_force
            kinetic = 0.5 * float(qd @ (Mr @ qd))
            elastic = 0.5 * float(q @ (Kr @ q))
            mechanical = kinetic + elastic + contact.energy
            history.append(
                FullFEMContactStep(
                    time=float(load_time),
                    min_gap=contact.min_gap,
                    max_penetration=contact.max_penetration,
                    normal_force=contact.normal_force,
                    kinetic_energy=kinetic,
                    elastic_energy=elastic,
                    contact_work=contact_work,
                    external_work=external_work,
                    energy_balance_error=mechanical - external_work,
                )
            )

        return FullFEMSimulationResult(history, basis @ q, basis @ qd, np.asarray(displacement_history), np.asarray(velocity_history))

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

    def _contact(self, displacement: np.ndarray) -> _QuadraturePlaneContact:
        return _surface_quadrature_plane_contact(
            self.fem.mesh.nodes,
            self._contact_faces,
            displacement,
            point=self.rigid_plane.point,
            normal=self.rigid_plane.normal,
            stiffness=self.contact_stiffness,
        )


class AdaptiveCalculixAlignedROMContactSimulator:
    """Adaptive patch-mode ROM using the aligned surface-quadrature contact law.

    The old adaptive simulator activates patch modes but evaluates nodal contact
    explicitly.  This solver keeps the validated CalculiX-aligned contact
    residual and tangent, while changing the reduced basis online from accepted
    contact quadrature points.  If the active set changes during a step, the
    step is retried from the previous accepted state with the enlarged basis.
    """

    def __init__(
        self,
        fem: FEMSystem,
        surface: SurfaceMesh,
        patches: list[SurfacePatch],
        base_basis: np.ndarray,
        patch_modes: dict[int, np.ndarray],
        rigid_plane: PrescribedRigidPlane,
        contact_stiffness: float,
        rayleigh_alpha: float = 0.0,
        rayleigh_beta: float = 0.0,
        external_force: np.ndarray | Callable[[float], np.ndarray] | None = None,
        hht_alpha: float = 0.0,
        max_iterations: int = 20,
        tolerance: float = 1.0e-11,
        activation_count: int = 1,
        activation_radius: float | None = None,
        retain_active_patches: bool = True,
        max_activation_iterations: int = 4,
        activation_normal_alignment: float = 0.5,
        residual_activation_count: int = 0,
        residual_activation_tolerance: float = 0.0,
    ) -> None:
        if contact_stiffness <= 0.0:
            raise ValueError("contact_stiffness must be positive")
        if not (-1.0 / 3.0 <= hht_alpha <= 0.0):
            raise ValueError("hht_alpha must lie in [-1/3, 0]")
        if activation_count < 1:
            raise ValueError("activation_count must be positive")
        if max_activation_iterations < 1:
            raise ValueError("max_activation_iterations must be positive")
        if not -1.0 <= activation_normal_alignment <= 1.0:
            raise ValueError("activation_normal_alignment must be between -1 and 1")
        if residual_activation_count < 0:
            raise ValueError("residual_activation_count cannot be negative")
        if residual_activation_tolerance < 0.0:
            raise ValueError("residual_activation_tolerance cannot be negative")
        base = np.asarray(base_basis, dtype=float)
        if base.ndim != 2 or base.shape[0] != fem.mesh.n_dofs:
            raise ValueError(f"base_basis must have shape ({fem.mesh.n_dofs}, n_modes)")
        if base.shape[1] == 0:
            raise ValueError("base_basis must contain at least one mode")

        checked_patch_modes: dict[int, np.ndarray] = {}
        for patch_id, mode in patch_modes.items():
            mode_array = np.asarray(mode, dtype=float).reshape(-1)
            if mode_array.shape != (fem.mesh.n_dofs,):
                raise ValueError(f"patch mode {patch_id} must have shape ({fem.mesh.n_dofs},)")
            checked_patch_modes[int(patch_id)] = mode_array

        self.fem = fem
        self.surface = surface
        self.patches = list(patches)
        self.base_basis = base
        self.patch_modes = checked_patch_modes
        self.rigid_plane = rigid_plane
        self.contact_stiffness = float(contact_stiffness)
        self.rayleigh_alpha = float(rayleigh_alpha)
        self.rayleigh_beta = float(rayleigh_beta)
        self.hht_alpha = float(hht_alpha)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.activation_count = int(activation_count)
        self.activation_radius = activation_radius
        self.retain_active_patches = bool(retain_active_patches)
        self.max_activation_iterations = int(max_activation_iterations)
        self.activation_normal_alignment = float(activation_normal_alignment)
        self.residual_activation_count = int(residual_activation_count)
        self.residual_activation_tolerance = float(residual_activation_tolerance)
        self.external_force = external_force
        if external_force is not None and not callable(external_force):
            external_force_array = np.asarray(external_force, dtype=float)
            if external_force_array.shape != (fem.mesh.n_dofs,):
                raise ValueError(f"external_force must have shape ({fem.mesh.n_dofs},)")
            self.external_force = external_force_array
        self._contact_faces = _plane_aligned_hex_boundary_faces(fem.mesh.nodes, fem.mesh.elements, rigid_plane.normal)
        if self._contact_faces.size == 0:
            raise ValueError("no exterior HEX8 faces align with the contact plane normal")

    def run(self, dt: float, steps: int) -> AdaptiveROMSimulationResult:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if steps < 1:
            raise ValueError("steps must be positive")

        alpha = self.hht_alpha
        beta = 0.25 * (1.0 - alpha) ** 2
        gamma = 0.5 - alpha

        u = np.zeros(self.fem.mesh.n_dofs, dtype=float)
        v = np.zeros_like(u)
        active_patch_ids: tuple[int, ...] = ()
        basis = self._basis_for(active_patch_ids)
        q, qd = self._project_state(basis, u, v)
        contact0 = self._contact(u)
        previous_external_force = self._external_force(0.0)
        qdd = self._initial_acceleration(basis, q, qd, previous_external_force, contact0)

        history: list[AdaptiveROMContactStep] = []
        displacement_history: list[np.ndarray] = []
        velocity_history: list[np.ndarray] = []
        external_work = 0.0
        contact_work = 0.0

        for step_id in range(steps):
            load_time = (step_id + 1) * dt
            previous_u = u.copy()
            previous_v = v.copy()
            previous_acceleration = basis @ qdd
            current_external_force = self._external_force(load_time)

            trial_active = active_patch_ids
            accepted: tuple[
                tuple[int, ...],
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                _QuadraturePlaneContact,
                int,
            ] | None = None
            for activation_iteration in range(1, self.max_activation_iterations + 1):
                trial_basis = self._basis_for(trial_active)
                trial_q, trial_qd = self._project_state(trial_basis, previous_u, previous_v)
                trial_qdd = self._project_vector(trial_basis, previous_acceleration)
                previous_contact = self._contact(previous_u)
                previous_static = self._static_residual(
                    trial_basis,
                    trial_q,
                    trial_qd,
                    previous_external_force,
                    previous_contact,
                )
                next_q, next_qd, next_qdd, next_contact = self._implicit_step(
                    trial_basis,
                    trial_q,
                    trial_qd,
                    trial_qdd,
                    previous_static,
                    current_external_force,
                    dt,
                    beta,
                    gamma,
                    alpha,
                )
                next_u = trial_basis @ next_q
                next_v = trial_basis @ next_qd
                full_residual = self._full_equilibrium_residual(
                    trial_basis,
                    next_q,
                    next_qd,
                    next_qdd,
                    current_external_force,
                    next_contact,
                )
                candidate_active = self._activate(next_contact.contact_points, trial_active, full_residual)
                accepted = (
                    candidate_active,
                    trial_basis,
                    next_q,
                    next_qd,
                    next_qdd,
                    next_u,
                    next_v,
                    next_contact,
                    activation_iteration,
                )
                if candidate_active == trial_active:
                    break
                trial_active = candidate_active

            if accepted is None:
                raise RuntimeError("adaptive ROM failed to produce a time-step candidate")

            (
                active_patch_ids,
                basis,
                q,
                qd,
                qdd,
                u,
                v,
                contact,
                used_activation_iterations,
            ) = accepted

            if active_patch_ids != trial_active:
                accepted_acceleration = basis @ qdd
                basis = self._basis_for(active_patch_ids)
                q, qd = self._project_state(basis, u, v)
                qdd = self._project_vector(basis, accepted_acceleration)

            displacement_history.append(u.copy())
            velocity_history.append(v.copy())
            displacement_increment = u - previous_u
            contact_work += float(contact.vector @ displacement_increment)
            external_work += float(0.5 * (previous_external_force + current_external_force) @ displacement_increment)
            previous_external_force = current_external_force
            Kr = np.asarray(basis.T @ (self.fem.K @ basis), dtype=float)
            Mr = np.asarray(basis.T @ (self.fem.M @ basis), dtype=float)
            kinetic = 0.5 * float(qd @ (Mr @ qd))
            elastic = 0.5 * float(q @ (Kr @ q))
            mechanical = kinetic + elastic + contact.energy
            history.append(
                AdaptiveROMContactStep(
                    time=float(load_time),
                    min_gap=contact.min_gap,
                    max_penetration=contact.max_penetration,
                    normal_force=contact.normal_force,
                    active_patch_ids=active_patch_ids,
                    basis_size=int(basis.shape[1]),
                    activation_iterations=int(used_activation_iterations),
                    kinetic_energy=kinetic,
                    elastic_energy=elastic,
                    contact_work=contact_work,
                    external_work=external_work,
                    energy_balance_error=mechanical - external_work,
                )
            )

        return AdaptiveROMSimulationResult(history, u, v, np.asarray(displacement_history), np.asarray(velocity_history))

    def _implicit_step(
        self,
        basis: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        previous_static: np.ndarray,
        current_external_force: np.ndarray,
        dt: float,
        beta: float,
        gamma: float,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, _QuadraturePlaneContact]:
        Kr = np.asarray(basis.T @ (self.fem.K @ basis), dtype=float)
        Mr = np.asarray(basis.T @ (self.fem.M @ basis), dtype=float)
        Cr = self.rayleigh_alpha * Mr + self.rayleigh_beta * Kr
        reduced_external = basis.T @ current_external_force
        q_predict = q + dt * qd + dt * dt * (0.5 - beta) * qdd
        qd_predict = qd + dt * (1.0 - gamma) * qdd
        c0 = 1.0 / (beta * dt * dt)
        c1 = gamma / (beta * dt)
        q_guess = q_predict.copy()

        contact = self._contact(basis @ q_guess)
        for _iteration in range(max(1, self.max_iterations)):
            u_guess = basis @ q_guess
            contact = self._contact(u_guess)
            a_guess = c0 * (q_guess - q_predict)
            v_guess = qd_predict + gamma * dt * a_guess
            static = Kr @ q_guess + Cr @ v_guess - reduced_external - basis.T @ contact.vector
            residual = Mr @ a_guess + (1.0 + alpha) * static - alpha * previous_static
            contact_tangent = np.asarray(basis.T @ (contact.tangent @ basis), dtype=float)
            tangent = Mr * c0 + (Kr + Cr * c1 + contact_tangent) * (1.0 + alpha)
            correction = la.solve(0.5 * (tangent + tangent.T), -residual, assume_a="sym")
            q_guess += correction
            if np.linalg.norm(correction) <= self.tolerance * max(1.0, np.linalg.norm(q_guess)):
                break

        q_next = q_guess
        qdd_next = c0 * (q_next - q_predict)
        qd_next = qd_predict + gamma * dt * qdd_next
        contact_next = self._contact(basis @ q_next)
        return q_next, qd_next, qdd_next, contact_next

    def _basis_for(self, active_patch_ids: tuple[int, ...]) -> np.ndarray:
        columns = [self.base_basis]
        for patch_id in active_patch_ids:
            mode = self.patch_modes.get(int(patch_id))
            if mode is not None:
                columns.append(mode.reshape(-1, 1))
        basis = mass_orthonormalize(self.fem.M, np.column_stack(columns))
        if basis.shape[1] == 0:
            raise RuntimeError("active reduced basis is empty")
        return basis

    def _activate(
        self,
        contact_points: np.ndarray,
        previous_active: tuple[int, ...],
        residual: np.ndarray | None,
    ) -> tuple[int, ...]:
        selected: list[int] = []
        points = np.asarray(contact_points, dtype=float).reshape((-1, 3))
        if points.size:
            distances: list[tuple[float, int]] = []
            for patch in self.patches:
                if float(np.dot(patch.normal, self.rigid_plane.normal)) < self.activation_normal_alignment:
                    continue
                point_distances = np.linalg.norm(points - patch.centroid, axis=1)
                distance = float(np.min(point_distances))
                if self.activation_radius is None or distance <= self.activation_radius:
                    distances.append((distance, int(patch.patch_id)))
            distances.sort()
            selected.extend(patch_id for _, patch_id in distances[: self.activation_count])
        if residual is not None and self.residual_activation_count > 0:
            selected.extend(self._residual_patch_candidates(residual, previous_active, tuple(selected)))
        if not self.retain_active_patches:
            return _unique_tuple(selected)
        retained = list(previous_active)
        for patch_id in selected:
            if patch_id not in retained:
                retained.append(patch_id)
        return tuple(retained)

    def _residual_patch_candidates(
        self,
        residual: np.ndarray,
        previous_active: tuple[int, ...],
        point_selected: tuple[int, ...],
    ) -> tuple[int, ...]:
        inactive = set(self.patch_modes) - set(previous_active) - set(point_selected)
        if not inactive:
            return ()
        residual_vector = np.asarray(residual, dtype=float)
        scores: list[tuple[float, int]] = []
        for patch_id in sorted(inactive):
            mode = self.patch_modes[int(patch_id)]
            mass_norm = float(np.sqrt(max(mode @ (self.fem.M @ mode), 0.0)))
            if mass_norm <= 1.0e-30:
                continue
            score = abs(float(mode @ residual_vector)) / mass_norm
            if score > self.residual_activation_tolerance:
                scores.append((score, int(patch_id)))
        scores.sort(reverse=True)
        return tuple(patch_id for _, patch_id in scores[: self.residual_activation_count])

    def _project_state(self, basis: np.ndarray, displacement: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Mr = np.asarray(basis.T @ (self.fem.M @ basis), dtype=float)
        q = la.solve(Mr, basis.T @ (self.fem.M @ displacement), assume_a="sym")
        qd = la.solve(Mr, basis.T @ (self.fem.M @ velocity), assume_a="sym")
        return q, qd

    def _project_vector(self, basis: np.ndarray, vector: np.ndarray) -> np.ndarray:
        Mr = np.asarray(basis.T @ (self.fem.M @ basis), dtype=float)
        return la.solve(Mr, basis.T @ (self.fem.M @ vector), assume_a="sym")

    def _initial_acceleration(
        self,
        basis: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        external_force: np.ndarray,
        contact: _QuadraturePlaneContact,
    ) -> np.ndarray:
        Kr = np.asarray(basis.T @ (self.fem.K @ basis), dtype=float)
        Mr = np.asarray(basis.T @ (self.fem.M @ basis), dtype=float)
        Cr = self.rayleigh_alpha * Mr + self.rayleigh_beta * Kr
        rhs = basis.T @ (external_force + contact.vector) - Kr @ q - Cr @ qd
        return la.solve(Mr, rhs, assume_a="sym")

    def _static_residual(
        self,
        basis: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        external_force: np.ndarray,
        contact: _QuadraturePlaneContact,
    ) -> np.ndarray:
        Kr = np.asarray(basis.T @ (self.fem.K @ basis), dtype=float)
        Mr = np.asarray(basis.T @ (self.fem.M @ basis), dtype=float)
        Cr = self.rayleigh_alpha * Mr + self.rayleigh_beta * Kr
        return Kr @ q + Cr @ qd - basis.T @ (external_force + contact.vector)

    def _full_equilibrium_residual(
        self,
        basis: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        external_force: np.ndarray,
        contact: _QuadraturePlaneContact,
    ) -> np.ndarray:
        displacement = basis @ q
        velocity = basis @ qd
        acceleration = basis @ qdd
        damping_force = self.rayleigh_alpha * (self.fem.M @ velocity) + self.rayleigh_beta * (self.fem.K @ velocity)
        return external_force + contact.vector - self.fem.M @ acceleration - damping_force - self.fem.K @ displacement

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

    def _contact(self, displacement: np.ndarray) -> _QuadraturePlaneContact:
        return _surface_quadrature_plane_contact(
            self.fem.mesh.nodes,
            self._contact_faces,
            displacement,
            point=self.rigid_plane.point,
            normal=self.rigid_plane.normal,
            stiffness=self.contact_stiffness,
        )


_HEX8_FACE_NODE_IDS = (
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
)
_QUAD_GAUSS = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))


def _plane_aligned_hex_boundary_faces(nodes: np.ndarray, elements: np.ndarray, normal: np.ndarray) -> np.ndarray:
    records: dict[tuple[int, ...], list[tuple[int, ...] | int]] = {}
    for element in np.asarray(elements, dtype=np.int64):
        for local_face in _HEX8_FACE_NODE_IDS:
            face = tuple(int(element[index]) for index in local_face)
            key = tuple(sorted(face))
            if key in records:
                records[key][1] = int(records[key][1]) + 1
            else:
                records[key] = [face, 1]

    mesh_center = np.mean(nodes, axis=0)
    target = np.asarray(normal, dtype=float)
    target /= max(float(np.linalg.norm(target)), 1.0e-30)
    faces: list[tuple[int, ...]] = []
    for face_value, count in records.values():
        if int(count) != 1:
            continue
        face = tuple(face_value)  # type: ignore[arg-type]
        coords = nodes[list(face)]
        face_normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        centroid = np.mean(coords, axis=0)
        if float(np.dot(face_normal, centroid - mesh_center)) < 0.0:
            face_normal = -face_normal
        norm = float(np.linalg.norm(face_normal))
        if norm <= 0.0:
            continue
        if float(np.dot(face_normal / norm, target)) >= 0.5:
            faces.append(face)
    return np.asarray(faces, dtype=np.int64)


def _quad_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = 0.25 * np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 - xi) * (1.0 + eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 + xi) * (1.0 - eta),
        ],
        dtype=float,
    )
    d_dxi = 0.25 * np.asarray([-(1.0 - eta), -(1.0 + eta), 1.0 + eta, 1.0 - eta], dtype=float)
    d_deta = 0.25 * np.asarray([-(1.0 - xi), 1.0 - xi, 1.0 + xi, -(1.0 + xi)], dtype=float)
    return shape, d_dxi, d_deta


def _surface_quadrature_plane_contact(
    nodes: np.ndarray,
    faces: np.ndarray,
    displacement: np.ndarray,
    *,
    point: np.ndarray,
    normal: np.ndarray,
    stiffness: float,
) -> _QuadraturePlaneContact:
    current = np.asarray(nodes, dtype=float) + np.asarray(displacement, dtype=float).reshape((-1, 3))
    plane_point = np.asarray(point, dtype=float)
    plane_normal = np.asarray(normal, dtype=float)
    plane_normal /= max(float(np.linalg.norm(plane_normal)), 1.0e-30)
    force_normal = -plane_normal
    vector = np.zeros(3 * current.shape[0], dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    min_gap = np.inf
    max_penetration = 0.0
    normal_force = 0.0
    energy = 0.0
    contact_nodes: set[int] = set()
    contact_points: list[np.ndarray] = []

    for face in np.asarray(faces, dtype=np.int64):
        face_coords = current[face]
        for xi in _QUAD_GAUSS:
            for eta in _QUAD_GAUSS:
                shape, d_dxi, d_deta = _quad_shape(float(xi), float(eta))
                point_q = shape @ face_coords
                tangent_xi = d_dxi @ face_coords
                tangent_eta = d_deta @ face_coords
                area = float(np.linalg.norm(np.cross(tangent_xi, tangent_eta)))
                if area <= 0.0:
                    continue
                signed = float((point_q - plane_point) @ plane_normal)
                gap = -signed
                min_gap = min(min_gap, gap)
                penetration = max(signed, 0.0)
                if penetration <= 0.0:
                    continue
                lam = float(stiffness) * area * penetration
                tangent_scale = float(stiffness) * area
                max_penetration = max(max_penetration, penetration)
                normal_force += lam
                energy += 0.5 * float(stiffness) * area * penetration * penetration
                contact_points.append(point_q.copy())
                contact_nodes.update(int(node_id) for node_id in face)
                for local_node, node_id in enumerate(face):
                    start = 3 * int(node_id)
                    vector[start : start + 3] += shape[local_node] * lam * force_normal
                for a, weight_a in enumerate(shape):
                    for b, weight_b in enumerate(shape):
                        block = weight_a * weight_b * tangent_scale * np.outer(plane_normal, plane_normal)
                        row_base = 3 * int(face[a])
                        col_base = 3 * int(face[b])
                        for i in range(3):
                            for j in range(3):
                                rows.append(row_base + i)
                                cols.append(col_base + j)
                                data.append(float(block[i, j]))

    if not np.isfinite(min_gap):
        min_gap = 0.0
    tangent = sp.coo_matrix((data, (rows, cols)), shape=(3 * current.shape[0], 3 * current.shape[0])).tocsr()
    return _QuadraturePlaneContact(
        vector=vector,
        tangent=tangent,
        min_gap=float(min_gap),
        max_penetration=float(max_penetration),
        normal_force=float(normal_force),
        energy=float(energy),
        contact_node_ids=np.asarray(sorted(contact_nodes), dtype=np.int64),
        contact_points=np.asarray(contact_points, dtype=float).reshape((-1, 3)),
    )


def _unique_tuple(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    unique: list[int] = []
    for value in values:
        item = int(value)
        if item not in unique:
            unique.append(item)
    return tuple(unique)
