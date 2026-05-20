from __future__ import annotations

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_ilc import (
    build_normal_patch_load_bases,
    detect_sdf_contact_samples,
    project_contact_samples_to_patch_ilc,
    total_nodal_force,
    total_nodal_moment,
    write_ilc_projection_tables,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def test_patch_ilc_projection_conserves_total_force_and_moment_for_distributed_contact() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    active_patches = _right_side_patches(level.patches)[:2]
    points, areas = _distributed_patch_contact_points(surface, active_patches, penetration=0.015)
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, level.patches)

    samples = detect_sdf_contact_samples(points, surface, level, penalty=200.0, sample_areas=areas)
    projection = project_contact_samples_to_patch_ilc(fem.mesh, surface, samples, load_bases)

    np.testing.assert_allclose(projection.projected_force_vector, projection.sample_lumped_force_vector, atol=1.0e-12)
    np.testing.assert_allclose(total_nodal_force(projection.projected_force_vector), samples.total_force, atol=1.0e-12)
    np.testing.assert_allclose(total_nodal_moment(fem.mesh, projection.projected_force_vector), samples.total_moment(), atol=1.0e-12)
    assert projection.total_force_error < 1.0e-12
    assert projection.total_normal_force_error < 1.0e-12
    assert projection.total_moment_error < 1.0e-12
    assert projection.total_normal_force_error < 0.01
    assert projection.total_moment_error < 0.05
    assert projection.active_patch_count == len(active_patches)
    assert projection.active_patch_count < len(level.patches) // 2


def test_ilc_projection_diagnostics_are_written(tmp_path) -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    active_patches = _right_side_patches(level.patches)[:2]
    points, areas = _distributed_patch_contact_points(surface, active_patches, penetration=0.015)
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, level.patches)
    samples = detect_sdf_contact_samples(points, surface, level, penalty=200.0, sample_areas=areas)
    projection = project_contact_samples_to_patch_ilc(fem.mesh, surface, samples, load_bases)

    write_ilc_projection_tables(projection, tmp_path)

    assert (tmp_path / "contact_to_patch_map.csv").read_text(encoding="utf-8").startswith("sample_id,triangle_id")
    assert "total_normal_force_error" in (tmp_path / "force_projection_error.csv").read_text(encoding="utf-8")
    assert "total_moment_error" in (tmp_path / "moment_projection_error.csv").read_text(encoding="utf-8")
    assert "ILC Projection Summary" in (tmp_path / "ilc_projection_summary.md").read_text(encoding="utf-8")


def _distributed_patch_contact_points(surface, patches, penetration: float) -> tuple[np.ndarray, np.ndarray]:
    points = []
    areas = []
    for patch in patches:
        for triangle_id in patch.triangle_indices:
            points.append(surface.centroids[triangle_id] - penetration * surface.normals[triangle_id])
            areas.append(surface.areas[triangle_id])
    return np.asarray(points, dtype=float), np.asarray(areas, dtype=float)


def _right_side_patches(patches):
    right = [patch for patch in patches if patch.key[0] == 0 and patch.key[1] == 1]
    assert len(right) >= 2
    return sorted(right, key=lambda patch: (patch.centroid[1], patch.centroid[2]))
