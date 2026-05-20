from __future__ import annotations

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.surface_patch import (
    build_patch_hierarchy,
    build_surface_samples,
    boundary_faces_to_surface_mesh,
    extract_boundary_faces,
    extract_surface,
    write_patch_hierarchy_tables,
)
from modal_contact_rom.surface_patch.patch_hierarchy import PatchLevel


def test_boundary_surface_sample_hierarchy_pipeline_covers_exterior_faces() -> None:
    mesh = cantilever_block(nx=2, ny=1, nz=1).mesh

    boundary_faces = extract_boundary_faces(mesh)
    surface_from_faces = boundary_faces_to_surface_mesh(mesh, boundary_faces)
    surface = extract_surface(mesh)
    samples = build_surface_samples(surface)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2))

    assert len(boundary_faces) == 10
    assert surface_from_faces.triangles.shape == surface.triangles.shape
    np.testing.assert_allclose(np.sum(surface_from_faces.areas), np.sum(surface.areas), atol=1.0e-12)
    assert samples.count == surface.triangles.shape[0]
    assert samples.points.shape == surface.centroids.shape
    assert hierarchy.level("coarse").coverage_ratio() == 1.0
    assert hierarchy.level("medium").coverage_ratio() == 1.0


def test_nonoverlap_patch_hierarchy_has_exact_coverage_adjacency_and_normals() -> None:
    surface = extract_surface(cantilever_block(nx=2, ny=2, nz=2).mesh)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2))

    for level_name in ("coarse", "medium"):
        level = hierarchy.level(level_name)
        row_sums = level.triangle_weights.sum(axis=1)
        nonzero_counts = np.count_nonzero(level.triangle_weights > 0.0, axis=1)

        assert level.coverage_ratio() == 1.0
        np.testing.assert_allclose(row_sums, 1.0, atol=1.0e-12)
        np.testing.assert_array_equal(nonzero_counts, np.ones(surface.triangles.shape[0], dtype=int))
        assert level.surface_area_error(surface) < 1.0e-10
        _assert_patch_normals_match_surface_average(surface, level)
        _assert_patch_adjacency_matches_shared_edges(surface, level)


def test_overlap_patch_weights_are_partition_of_unity() -> None:
    surface = extract_surface(cantilever_block(nx=2, ny=2, nz=2).mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium_overlap")

    row_sums = level.triangle_weights.sum(axis=1)
    nonzero_counts = np.count_nonzero(level.triangle_weights > 0.0, axis=1)

    assert level.overlap
    assert level.coverage_ratio() == 1.0
    np.testing.assert_allclose(row_sums, 1.0, atol=1.0e-12)
    assert np.max(nonzero_counts) > 1
    assert level.surface_area_error(surface) < 1.0e-10
    assert np.all(level.weighted_patch_areas(surface) > 0.0)


def test_patch_hierarchy_accepts_irregular_hex_surface_and_writes_diagnostics(tmp_path) -> None:
    base = cantilever_block(nx=2, ny=1, nz=1).mesh
    nodes = base.nodes.copy()
    nodes[:, 1] += 0.08 * np.sin(1.7 * nodes[:, 0] + 0.3 * nodes[:, 2])
    nodes[:, 2] += 0.05 * np.cos(1.1 * nodes[:, 0] - 0.4 * nodes[:, 1])
    mesh = Mesh(nodes=nodes, elements=base.elements, element_type=base.element_type)
    surface = extract_surface(mesh)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 1))

    for level in hierarchy.levels:
        np.testing.assert_allclose(level.triangle_weights.sum(axis=1), 1.0, atol=1.0e-12)
        assert level.coverage_ratio() == 1.0
        assert level.surface_area_error(surface) < 1.0e-8
        _assert_patch_normals_match_surface_average(surface, level)

    write_patch_hierarchy_tables(hierarchy, tmp_path)
    assert (tmp_path / "patch_stats.csv").read_text(encoding="utf-8").startswith("level,patch_id")
    assert "coverage_ratio" in (tmp_path / "patch_coverage.csv").read_text(encoding="utf-8")
    assert "Patch Hierarchy Summary" in (tmp_path / "patch_summary.md").read_text(encoding="utf-8")


def _assert_patch_normals_match_surface_average(surface, level: PatchLevel) -> None:
    for patch in level.patches:
        areas = surface.areas[patch.triangle_indices]
        average_normal = np.average(surface.normals[patch.triangle_indices], axis=0, weights=areas)
        average_normal /= np.linalg.norm(average_normal)
        assert float(np.dot(patch.normal, average_normal)) > 0.999999


def _assert_patch_adjacency_matches_shared_edges(surface, level: PatchLevel) -> None:
    expected = _expected_adjacency_from_shared_edges(surface, level.primary_patch_ids)
    assert level.adjacency == expected
    assert any(neighbors for neighbors in level.adjacency.values())
    for patch_id, neighbors in level.adjacency.items():
        for neighbor in neighbors:
            assert patch_id in level.adjacency[neighbor]


def _expected_adjacency_from_shared_edges(surface, primary_patch_ids: np.ndarray) -> dict[int, tuple[int, ...]]:
    neighbors: dict[int, set[int]] = {int(patch_id): set() for patch_id in np.unique(primary_patch_ids)}
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for triangle_id, triangle in enumerate(surface.triangles):
        edges = ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))
        for a, b in edges:
            edge_to_triangles.setdefault(tuple(sorted((int(a), int(b)))), []).append(triangle_id)

    for triangle_ids in edge_to_triangles.values():
        patch_ids = sorted({int(primary_patch_ids[triangle_id]) for triangle_id in triangle_ids})
        for patch_id in patch_ids:
            neighbors[patch_id].update(other for other in patch_ids if other != patch_id)
    return {patch_id: tuple(sorted(patch_neighbors)) for patch_id, patch_neighbors in neighbors.items()}
