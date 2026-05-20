from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modal_contact_rom.surface_patch.extract import SurfaceMesh


@dataclass(frozen=True)
class SignedDistanceResult:
    distances: np.ndarray
    closest_points: np.ndarray
    triangle_indices: np.ndarray


def query_signed_distances(points: np.ndarray, surface: SurfaceMesh) -> SignedDistanceResult:
    """Approximate signed distances to the extracted triangle surface.

    The sign is based on the closest triangle normal. It is adequate for local
    contact queries in this prototype, but not a robust closed-mesh winding test.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")

    distances = np.empty(points.shape[0], dtype=float)
    closest_points = np.empty_like(points)
    triangle_indices = np.empty(points.shape[0], dtype=np.int64)

    tris = surface.nodes[surface.triangles]
    for point_id, point in enumerate(points):
        best_dist2 = np.inf
        best_point = None
        best_tri = -1
        for tri_id, triangle in enumerate(tris):
            candidate = _closest_point_on_triangle(point, triangle[0], triangle[1], triangle[2])
            dist2 = float(np.sum((point - candidate) ** 2))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_point = candidate
                best_tri = tri_id
        assert best_point is not None
        sign = 1.0 if np.dot(point - best_point, surface.normals[best_tri]) >= 0.0 else -1.0
        distances[point_id] = sign * float(np.sqrt(best_dist2))
        closest_points[point_id] = best_point
        triangle_indices[point_id] = best_tri

    return SignedDistanceResult(distances, closest_points, triangle_indices)


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w

