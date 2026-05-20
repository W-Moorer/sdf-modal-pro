from __future__ import annotations

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.sdf_query.mesh_distance import _closest_point_on_triangle, query_signed_distances
from modal_contact_rom.surface_patch import extract_surface


def test_vectorized_sdf_query_matches_scalar_triangle_search() -> None:
    fem = cantilever_block(nx=2, ny=3, nz=2, length=2.0, width=1.3, height=0.8)
    nodes = fem.mesh.nodes.copy()
    nodes[:, 0] += 0.08 * np.sin(2.0 * np.pi * nodes[:, 1]) * np.cos(np.pi * nodes[:, 2])
    mesh = Mesh(nodes, fem.mesh.elements, fem.mesh.element_type)
    surface = extract_surface(mesh)

    triangle_ids = np.arange(0, surface.triangles.shape[0], 3, dtype=np.int64)[:8]
    points = []
    for tri_id in triangle_ids:
        points.append(surface.centroids[tri_id] - 0.013 * surface.normals[tri_id])
        points.append(surface.centroids[tri_id] + 0.017 * surface.normals[tri_id])
    points_array = np.asarray(points, dtype=float)

    result = query_signed_distances(points_array, surface)
    scalar_distances = []
    scalar_closest = []
    scalar_triangle_ids = []
    triangles = surface.nodes[surface.triangles]
    for point in points_array:
        best_distance = np.inf
        best_point = None
        best_triangle_id = -1
        for tri_id, triangle in enumerate(triangles):
            candidate = _closest_point_on_triangle(point, triangle[0], triangle[1], triangle[2])
            distance = float(np.sum((point - candidate) ** 2))
            if distance < best_distance:
                best_distance = distance
                best_point = candidate
                best_triangle_id = tri_id
        assert best_point is not None
        sign = 1.0 if np.dot(point - best_point, surface.normals[best_triangle_id]) >= 0.0 else -1.0
        scalar_distances.append(sign * float(np.sqrt(best_distance)))
        scalar_closest.append(best_point)
        scalar_triangle_ids.append(best_triangle_id)

    np.testing.assert_allclose(result.distances, np.asarray(scalar_distances), rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(result.closest_points, np.asarray(scalar_closest), rtol=0.0, atol=1.0e-13)
    np.testing.assert_array_equal(result.triangle_indices, np.asarray(scalar_triangle_ids, dtype=np.int64))
