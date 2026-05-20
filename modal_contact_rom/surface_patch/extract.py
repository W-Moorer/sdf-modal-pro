from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from modal_contact_rom.fem_io.mesh import Mesh


@dataclass(frozen=True)
class SurfaceMesh:
    nodes: np.ndarray
    triangles: np.ndarray
    areas: np.ndarray
    normals: np.ndarray
    centroids: np.ndarray


@dataclass(frozen=True)
class SurfacePatch:
    patch_id: int
    key: tuple[int, int, int, int]
    triangle_indices: np.ndarray
    node_indices: np.ndarray
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


def extract_surface(mesh: Mesh) -> SurfaceMesh:
    """Extract oriented exterior triangles from tet4 or hex8 elements."""

    faces = _HEX_FACES if mesh.element_type == "hex8" else _TET_FACES
    seen: dict[tuple[int, ...], list[tuple[int, ...] | int]] = {}

    for element in mesh.elements:
        for face in faces:
            face_nodes = tuple(int(element[i]) for i in face)
            key = tuple(sorted(face_nodes))
            if key in seen:
                seen[key][1] = int(seen[key][1]) + 1
            else:
                seen[key] = [face_nodes, 1]

    center = np.mean(mesh.nodes, axis=0)
    triangles: list[tuple[int, int, int]] = []
    for face_nodes, count in seen.values():
        if int(count) != 1:
            continue
        face_tuple = tuple(face_nodes)  # type: ignore[arg-type]
        if len(face_tuple) == 3:
            triangles.append(_oriented_triangle(mesh.nodes, face_tuple, center))
        else:
            a, b, c, d = face_tuple
            triangles.append(_oriented_triangle(mesh.nodes, (a, b, c), center))
            triangles.append(_oriented_triangle(mesh.nodes, (a, c, d), center))

    tri_array = np.asarray(triangles, dtype=np.int64)
    areas, normals, centroids = _triangle_geometry(mesh.nodes, tri_array)
    return SurfaceMesh(mesh.nodes, tri_array, areas, normals, centroids)


def partition_surface_patches(
    surface: SurfaceMesh,
    bins: tuple[int, int] = (2, 2),
) -> list[SurfacePatch]:
    """Partition each major surface side into a regular tangent grid."""

    if min(bins) < 1:
        raise ValueError("patch bins must be positive")

    side_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for tri_id, normal in enumerate(surface.normals):
        axis = int(np.argmax(np.abs(normal)))
        sign = 1 if normal[axis] >= 0.0 else -1
        side_to_indices[(axis, sign)].append(tri_id)

    keyed_triangles: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for side, tri_indices in side_to_indices.items():
        axis, sign = side
        tangent_axes = [i for i in range(3) if i != axis]
        coords = surface.centroids[np.asarray(tri_indices)][:, tangent_axes]
        mins = coords.min(axis=0)
        spans = np.maximum(coords.max(axis=0) - mins, 1.0e-12)
        for tri_id in tri_indices:
            uv = (surface.centroids[tri_id, tangent_axes] - mins) / spans
            iu = min(int(np.floor(uv[0] * bins[0])), bins[0] - 1)
            iv = min(int(np.floor(uv[1] * bins[1])), bins[1] - 1)
            keyed_triangles[(axis, sign, iu, iv)].append(tri_id)

    patches: list[SurfacePatch] = []
    for patch_id, key in enumerate(sorted(keyed_triangles)):
        tri_indices = np.asarray(keyed_triangles[key], dtype=np.int64)
        node_indices = np.unique(surface.triangles[tri_indices].reshape(-1))
        weights = surface.areas[tri_indices]
        area = float(np.sum(weights))
        normal = np.average(surface.normals[tri_indices], axis=0, weights=weights)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 0.0:
            normal = normal / normal_norm
        centroid = np.average(surface.centroids[tri_indices], axis=0, weights=weights)
        patches.append(SurfacePatch(patch_id, key, tri_indices, node_indices, area, normal, centroid))
    return patches


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

