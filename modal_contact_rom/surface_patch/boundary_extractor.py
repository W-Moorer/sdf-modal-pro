from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.surface_patch.extract import SurfaceMesh


@dataclass(frozen=True)
class BoundaryFace:
    face_id: int
    element_id: int
    local_face_id: int
    node_indices: tuple[int, ...]
    triangles: tuple[tuple[int, int, int], ...]
    area: float
    normal: np.ndarray
    centroid: np.ndarray


_HEX_FACES = (
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
)

_TET_FACES = (
    (0, 2, 1),
    (0, 1, 3),
    (1, 2, 3),
    (2, 0, 3),
)


def extract_boundary_faces(mesh: Mesh) -> list[BoundaryFace]:
    """Extract oriented exterior faces from a tet4 or hex8 volume mesh."""

    local_faces = _HEX_FACES if mesh.element_type == "hex8" else _TET_FACES
    seen: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
    for element_id, element in enumerate(mesh.elements):
        for local_face_id, local_face in enumerate(local_faces):
            face_nodes = tuple(int(element[index]) for index in local_face)
            key = tuple(sorted(face_nodes))
            seen.setdefault(key, []).append((element_id, local_face_id, face_nodes))

    center = np.mean(mesh.nodes, axis=0)
    boundary_faces: list[BoundaryFace] = []
    for records in seen.values():
        if len(records) != 1:
            continue
        element_id, local_face_id, face_nodes = records[0]
        triangles = _triangulate_oriented_face(mesh.nodes, face_nodes, center)
        areas, normals, centroids = _triangle_geometry(mesh.nodes, np.asarray(triangles, dtype=np.int64))
        area = float(np.sum(areas))
        normal = np.average(normals, axis=0, weights=np.maximum(areas, 1.0e-30))
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 0.0:
            normal = normal / normal_norm
        centroid = np.average(centroids, axis=0, weights=np.maximum(areas, 1.0e-30))
        boundary_faces.append(
            BoundaryFace(
                face_id=len(boundary_faces),
                element_id=int(element_id),
                local_face_id=int(local_face_id),
                node_indices=tuple(int(node_id) for node_id in face_nodes),
                triangles=tuple(triangles),
                area=area,
                normal=normal,
                centroid=centroid,
            )
        )
    return boundary_faces


def boundary_faces_to_surface_mesh(mesh: Mesh, boundary_faces: list[BoundaryFace]) -> SurfaceMesh:
    """Convert extracted boundary faces into the triangle-based surface mesh."""

    triangles = [triangle for face in boundary_faces for triangle in face.triangles]
    tri_array = np.asarray(triangles, dtype=np.int64)
    areas, normals, centroids = _triangle_geometry(mesh.nodes, tri_array)
    return SurfaceMesh(mesh.nodes, tri_array, areas, normals, centroids)


def _triangulate_oriented_face(
    nodes: np.ndarray,
    face_nodes: tuple[int, ...],
    center: np.ndarray,
) -> tuple[tuple[int, int, int], ...]:
    if len(face_nodes) == 3:
        return (_oriented_triangle(nodes, face_nodes, center),)
    if len(face_nodes) == 4:
        a, b, c, d = face_nodes
        return (
            _oriented_triangle(nodes, (a, b, c), center),
            _oriented_triangle(nodes, (a, c, d), center),
        )
    raise ValueError("only triangular and quadrilateral boundary faces are supported")


def _oriented_triangle(nodes: np.ndarray, triangle: tuple[int, int, int], center: np.ndarray) -> tuple[int, int, int]:
    a, b, c = triangle
    pa, pb, pc = nodes[[a, b, c]]
    normal = np.cross(pb - pa, pc - pa)
    centroid = (pa + pb + pc) / 3.0
    if np.dot(normal, centroid - center) < 0.0:
        return (a, c, b)
    return triangle


def _triangle_geometry(nodes: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pa = nodes[triangles[:, 0]]
    pb = nodes[triangles[:, 1]]
    pc = nodes[triangles[:, 2]]
    raw_normals = np.cross(pb - pa, pc - pa)
    raw_lengths = np.linalg.norm(raw_normals, axis=1)
    areas = 0.5 * raw_lengths
    normals = np.zeros_like(raw_normals)
    valid = raw_lengths > 0.0
    normals[valid] = raw_normals[valid] / raw_lengths[valid, None]
    centroids = (pa + pb + pc) / 3.0
    return areas, normals, centroids
