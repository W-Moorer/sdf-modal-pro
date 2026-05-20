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
    if points.shape[0] == 0:
        return SignedDistanceResult(distances, closest_points, triangle_indices)

    tris = surface.nodes[surface.triangles]
    a = tris[:, 0]
    b = tris[:, 1]
    c = tris[:, 2]
    candidates = _closest_points_on_triangle_batch(points, a, b, c)
    deltas = points[:, None, :] - candidates
    dist2 = np.einsum("ijk,ijk->ij", deltas, deltas)
    triangle_indices[:] = np.argmin(dist2, axis=1)
    point_ids = np.arange(points.shape[0], dtype=np.int64)
    closest_points[:] = candidates[point_ids, triangle_indices]
    signed_offsets = np.einsum("ij,ij->i", points - closest_points, surface.normals[triangle_indices])
    signs = np.where(signed_offsets >= 0.0, 1.0, -1.0)
    distances[:] = signs * np.sqrt(dist2[point_ids, triangle_indices])

    return SignedDistanceResult(distances, closest_points, triangle_indices)


def _closest_points_on_triangle_batch(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    point_count = points.shape[0]
    triangle_count = a.shape[0]
    closest = np.empty((point_count, triangle_count, 3), dtype=float)
    assigned = np.zeros((point_count, triangle_count), dtype=bool)

    a3 = a[None, :, :]
    b3 = b[None, :, :]
    c3 = c[None, :, :]
    ab3 = ab[None, :, :]
    ac3 = ac[None, :, :]
    p3 = points[:, None, :]

    ap = p3 - a3
    d1 = np.sum(ab3 * ap, axis=2)
    d2 = np.sum(ac3 * ap, axis=2)
    mask = (d1 <= 0.0) & (d2 <= 0.0)
    _assign_triangle_vertices(closest, assigned, mask, a)

    bp = p3 - b3
    d3 = np.sum(ab3 * bp, axis=2)
    d4 = np.sum(ac3 * bp, axis=2)
    mask = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
    _assign_triangle_vertices(closest, assigned, mask, b)

    vc = d1 * d4 - d3 * d2
    mask = (~assigned) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    if np.any(mask):
        rows, cols = np.nonzero(mask)
        v = d1[rows, cols] / (d1[rows, cols] - d3[rows, cols])
        closest[rows, cols] = a[cols] + v[:, None] * ab[cols]
        assigned[rows, cols] = True

    cp = p3 - c3
    d5 = np.sum(ab3 * cp, axis=2)
    d6 = np.sum(ac3 * cp, axis=2)
    mask = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
    _assign_triangle_vertices(closest, assigned, mask, c)

    vb = d5 * d2 - d1 * d6
    mask = (~assigned) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    if np.any(mask):
        rows, cols = np.nonzero(mask)
        w = d2[rows, cols] / (d2[rows, cols] - d6[rows, cols])
        closest[rows, cols] = a[cols] + w[:, None] * ac[cols]
        assigned[rows, cols] = True

    va = d3 * d6 - d5 * d4
    mask = (~assigned) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    if np.any(mask):
        rows, cols = np.nonzero(mask)
        w = (d4[rows, cols] - d3[rows, cols]) / (
            (d4[rows, cols] - d3[rows, cols]) + (d5[rows, cols] - d6[rows, cols])
        )
        closest[rows, cols] = b[cols] + w[:, None] * (c[cols] - b[cols])
        assigned[rows, cols] = True

    mask = ~assigned
    if np.any(mask):
        rows, cols = np.nonzero(mask)
        denom = 1.0 / (va[rows, cols] + vb[rows, cols] + vc[rows, cols])
        v = vb[rows, cols] * denom
        w = vc[rows, cols] * denom
        closest[rows, cols] = a[cols] + ab[cols] * v[:, None] + ac[cols] * w[:, None]

    return closest


def _assign_triangle_vertices(
    closest: np.ndarray,
    assigned: np.ndarray,
    mask: np.ndarray,
    vertices: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    rows, cols = np.nonzero(mask)
    closest[rows, cols] = vertices[cols]
    assigned[rows, cols] = True


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
