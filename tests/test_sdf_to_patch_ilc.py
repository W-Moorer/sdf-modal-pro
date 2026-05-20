from __future__ import annotations

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_ilc import (
    build_normal_patch_load_bases,
    detect_sdf_contact_samples,
    project_contact_samples_to_patch_ilc,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def test_single_sdf_contact_point_maps_to_expected_patch() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    patch = _right_side_patches(level.patches)[0]
    point = patch.centroid - 0.02 * patch.normal

    samples = detect_sdf_contact_samples(
        point[None, :],
        surface,
        level,
        penalty=100.0,
        sample_areas=np.array([patch.area]),
    )

    assert samples.sample_count == 1
    assert samples.active_patch_ids == (patch.patch_id,)
    assert int(samples.patch_ids[0]) == patch.patch_id
    assert samples.normal_forces[0] > 0.0


def test_multiple_sdf_contact_points_activate_multiple_patch_ilcs() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    patches = _right_side_patches(level.patches)[:2]
    points = np.asarray([patch.centroid - 0.02 * patch.normal for patch in patches])
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, level.patches)

    samples = detect_sdf_contact_samples(
        points,
        surface,
        level,
        penalty=100.0,
        sample_areas=np.asarray([patch.area for patch in patches]),
    )
    projection = project_contact_samples_to_patch_ilc(fem.mesh, surface, samples, load_bases)

    assert set(samples.active_patch_ids) == {patch.patch_id for patch in patches}
    assert set(projection.active_set.active_patch_ids) == {patch.patch_id for patch in patches}
    assert projection.active_patch_count == 2
    assert projection.active_patch_count < len(level.patches) // 2
    assert np.all(projection.active_set.alpha > 0.0)


def test_no_sdf_contact_returns_empty_alpha() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    patch = _right_side_patches(level.patches)[0]
    point = patch.centroid + 0.02 * patch.normal
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, level.patches)

    samples = detect_sdf_contact_samples(point[None, :], surface, level, penalty=100.0)
    projection = project_contact_samples_to_patch_ilc(fem.mesh, surface, samples, load_bases)

    assert samples.sample_count == 0
    assert projection.active_set.is_empty
    assert projection.active_set.alpha_dimension == 0
    assert projection.active_set.dynamic_dimension_contribution == 0
    np.testing.assert_allclose(projection.projected_force_vector, 0.0)


def _right_side_patches(patches):
    right = [patch for patch in patches if patch.key[0] == 0 and patch.key[1] == 1]
    assert len(right) >= 2
    return sorted(right, key=lambda patch: (patch.centroid[1], patch.centroid[2]))
