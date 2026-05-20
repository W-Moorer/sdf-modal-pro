from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

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
