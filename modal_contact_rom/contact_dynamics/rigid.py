from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.surface_patch.extract import SurfaceMesh


@dataclass(frozen=True)
class ContactEvaluation:
    gaps: np.ndarray
    normals: np.ndarray
    contact_node_ids: np.ndarray
    contact_points: np.ndarray
    min_gap: float


@dataclass(frozen=True)
class SurfaceContactForce:
    vector: np.ndarray
    resultant: np.ndarray
    normal_force: float
    max_penetration: float
    contact_node_ids: np.ndarray
    contact_points: np.ndarray


class PrescribedRigidSphere:
    """Rigid sphere with a prescribed center trajectory."""

    def __init__(
        self,
        radius: float,
        center: np.ndarray | Callable[[float], np.ndarray],
        velocity: np.ndarray | Callable[[float], np.ndarray] | None = None,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        self.radius = float(radius)
        self._center = center
        self._velocity = velocity

    def center(self, time: float) -> np.ndarray:
        if callable(self._center):
            return np.asarray(self._center(time), dtype=float)
        return np.asarray(self._center, dtype=float)

    def velocity(self, time: float) -> np.ndarray:
        if callable(self._velocity):
            return np.asarray(self._velocity(time), dtype=float)
        if self._velocity is not None:
            return np.asarray(self._velocity, dtype=float)
        return np.zeros(3, dtype=float)

    def evaluate(self, mesh: Mesh, surface: SurfaceMesh, displacement: np.ndarray, time: float) -> ContactEvaluation:
        center = self.center(time)
        surface_nodes = np.unique(surface.triangles.reshape(-1))
        current = mesh.nodes[surface_nodes] + displacement.reshape(-1, 3)[surface_nodes]
        offsets = current - center
        distances = np.linalg.norm(offsets, axis=1)
        normals = np.zeros_like(offsets)
        valid = distances > 1.0e-14
        normals[valid] = offsets[valid] / distances[valid, None]
        normals[~valid] = np.array([1.0, 0.0, 0.0])
        gaps = distances - self.radius
        contact_mask = gaps < 0.0
        contact_node_ids = surface_nodes[contact_mask]
        contact_points = current[contact_mask]
        return ContactEvaluation(
            gaps=gaps,
            normals=normals,
            contact_node_ids=contact_node_ids,
            contact_points=contact_points,
            min_gap=float(np.min(gaps)) if gaps.size else float("inf"),
        )

    def contact_force(
        self,
        mesh: Mesh,
        surface: SurfaceMesh,
        displacement: np.ndarray,
        velocity: np.ndarray,
        time: float,
        penalty: float,
        damping: float = 0.0,
    ) -> SurfaceContactForce:
        evaluation = self.evaluate(mesh, surface, displacement, time)
        vector = np.zeros(mesh.n_dofs, dtype=float)
        resultant = np.zeros(3, dtype=float)
        if evaluation.contact_node_ids.size == 0:
            return SurfaceContactForce(
                vector,
                resultant,
                normal_force=0.0,
                max_penetration=0.0,
                contact_node_ids=evaluation.contact_node_ids,
                contact_points=evaluation.contact_points,
            )

        sphere_velocity = self.velocity(time)
        surface_nodes = np.unique(surface.triangles.reshape(-1))
        contact_lookup = {int(node_id): idx for idx, node_id in enumerate(surface_nodes)}
        nodal_velocity = velocity.reshape(-1, 3)
        normal_force = 0.0
        max_penetration = 0.0
        for node_id in evaluation.contact_node_ids:
            local_idx = contact_lookup[int(node_id)]
            normal = evaluation.normals[local_idx]
            penetration = max(-float(evaluation.gaps[local_idx]), 0.0)
            relative_normal_speed = float(np.dot(nodal_velocity[int(node_id)] - sphere_velocity, normal))
            force_magnitude = penalty * penetration - damping * min(relative_normal_speed, 0.0)
            force_magnitude = max(force_magnitude, 0.0)
            force = force_magnitude * normal
            start = 3 * int(node_id)
            vector[start : start + 3] += force
            resultant += force
            normal_force += force_magnitude
            max_penetration = max(max_penetration, penetration)

        return SurfaceContactForce(
            vector,
            resultant,
            normal_force=float(normal_force),
            max_penetration=float(max_penetration),
            contact_node_ids=evaluation.contact_node_ids,
            contact_points=evaluation.contact_points,
        )


class PrescribedRigidPlane:
    """Rigid plane with penalty response on the positive-normal side."""

    def __init__(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        velocity: np.ndarray | Callable[[float], np.ndarray] | None = None,
        area_weighted: bool = False,
        area_normal_alignment: float = 0.5,
    ) -> None:
        self.point = np.asarray(point, dtype=float)
        normal = np.asarray(normal, dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("normal must be nonzero")
        self.normal = normal / normal_norm
        self._velocity = velocity
        self.area_weighted = bool(area_weighted)
        self.area_normal_alignment = float(area_normal_alignment)
        if not -1.0 <= self.area_normal_alignment <= 1.0:
            raise ValueError("area_normal_alignment must be between -1 and 1")

    def velocity(self, time: float) -> np.ndarray:
        if callable(self._velocity):
            return np.asarray(self._velocity(time), dtype=float)
        if self._velocity is not None:
            return np.asarray(self._velocity, dtype=float)
        return np.zeros(3, dtype=float)

    def evaluate(self, mesh: Mesh, surface: SurfaceMesh, displacement: np.ndarray, time: float) -> ContactEvaluation:
        del time
        surface_nodes = np.unique(surface.triangles.reshape(-1))
        current = mesh.nodes[surface_nodes] + displacement.reshape(-1, 3)[surface_nodes]
        signed = (current - self.point) @ self.normal
        gaps = -signed
        contact_mask = signed > 0.0
        normals = np.repeat((-self.normal)[None, :], surface_nodes.size, axis=0)
        return ContactEvaluation(
            gaps=gaps,
            normals=normals,
            contact_node_ids=surface_nodes[contact_mask],
            contact_points=current[contact_mask],
            min_gap=float(np.min(gaps)) if gaps.size else float("inf"),
        )

    def contact_force(
        self,
        mesh: Mesh,
        surface: SurfaceMesh,
        displacement: np.ndarray,
        velocity: np.ndarray,
        time: float,
        penalty: float,
        damping: float = 0.0,
    ) -> SurfaceContactForce:
        evaluation = self.evaluate(mesh, surface, displacement, time)
        vector = np.zeros(mesh.n_dofs, dtype=float)
        resultant = np.zeros(3, dtype=float)
        if evaluation.contact_node_ids.size == 0:
            return SurfaceContactForce(
                vector,
                resultant,
                normal_force=0.0,
                max_penetration=0.0,
                contact_node_ids=evaluation.contact_node_ids,
                contact_points=evaluation.contact_points,
            )

        plane_velocity = self.velocity(time)
        nodal_velocity = velocity.reshape(-1, 3)
        nodal_area = self.nodal_area_scales(mesh, surface)
        normal_force = 0.0
        max_penetration = 0.0
        for node_id, point in zip(evaluation.contact_node_ids, evaluation.contact_points):
            signed = float((point - self.point) @ self.normal)
            penetration = max(signed, 0.0)
            force_normal = -self.normal
            relative_normal_speed = float(np.dot(nodal_velocity[int(node_id)] - plane_velocity, force_normal))
            area_scale = float(nodal_area[int(node_id)])
            force_magnitude = area_scale * (penalty * penetration - damping * min(relative_normal_speed, 0.0))
            force_magnitude = max(force_magnitude, 0.0)
            force = force_magnitude * force_normal
            start = 3 * int(node_id)
            vector[start : start + 3] += force
            resultant += force
            normal_force += force_magnitude
            max_penetration = max(max_penetration, penetration)

        return SurfaceContactForce(
            vector,
            resultant,
            normal_force=float(normal_force),
            max_penetration=float(max_penetration),
            contact_node_ids=evaluation.contact_node_ids,
            contact_points=evaluation.contact_points,
        )

    def nodal_area_scales(self, mesh: Mesh, surface: SurfaceMesh) -> np.ndarray:
        if not self.area_weighted:
            return np.ones(mesh.n_nodes, dtype=float)
        areas = np.zeros(mesh.n_nodes, dtype=float)
        alignment = surface.normals @ self.normal
        triangle_ids = np.flatnonzero(alignment >= self.area_normal_alignment)
        for tri_id in triangle_ids:
            share = float(surface.areas[tri_id]) / 3.0
            for node_id in surface.triangles[tri_id]:
                areas[int(node_id)] += share
        return areas
