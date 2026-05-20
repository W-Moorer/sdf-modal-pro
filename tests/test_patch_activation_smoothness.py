from __future__ import annotations

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_ilc import (
    AdaptivePatchActivationConfig,
    AdaptivePatchILCProjector,
    build_normal_patch_load_bases,
    expand_patch_ids_with_neighbors,
    run_adaptive_patch_sequence,
    write_adaptive_patch_tables,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface
from modal_contact_rom.modal_ilc.sdf_projection import ContactSampleSet, detect_sdf_contact_samples


def test_patch_activation_includes_neighbors_and_delays_deactivation() -> None:
    case = _phase7_case()
    config = AdaptivePatchActivationConfig(neighbor_depth=1, deactivate_delay=2, alpha_blend=0.3)
    projector = AdaptivePatchILCProjector(case["fem"].mesh, case["surface"], case["medium"], case["medium_bases"], config)
    patch = _right_side_patches(case["medium"].patches)[0]
    points, areas = _single_patch_contact_frame(case["surface"], patch, penetration=0.01)
    samples = detect_sdf_contact_samples(points, case["surface"], case["medium"], penalty=100.0, sample_areas=areas)

    first = projector.step(samples, step_id=0)
    expected = expand_patch_ids_with_neighbors(case["medium"], first.requested_patch_ids, depth=1)
    assert set(first.requested_patch_ids).issubset(set(first.active_set.active_patch_ids))
    assert set(expected).issubset(set(first.active_set.active_patch_ids))

    no_contact = ContactSampleSet.empty()
    retained_1 = projector.step(no_contact, step_id=1)
    retained_2 = projector.step(no_contact, step_id=2)
    released = projector.step(no_contact, step_id=3)

    assert set(first.active_set.active_patch_ids).issubset(set(retained_1.active_set.active_patch_ids))
    assert set(first.active_set.active_patch_ids).issubset(set(retained_2.active_set.active_patch_ids))
    assert released.active_set.is_empty


def test_sliding_contact_patch_activation_is_smooth_and_more_local_than_coarse_only() -> None:
    case = _phase7_case()
    config = AdaptivePatchActivationConfig(
        neighbor_depth=1,
        deactivate_delay=4,
        alpha_blend=0.3,
        max_active_patches=8,
    )
    result = run_adaptive_patch_sequence(
        case["fem"].mesh,
        case["surface"],
        case["medium"],
        case["medium_bases"],
        _sliding_frames(case["surface"], steps=41),
        penalty=100.0,
        config=config,
        coarse_patch_level=case["coarse"],
        coarse_load_bases=case["coarse_bases"],
    )

    requested_history = [step.requested_patch_ids for step in result.steps]
    assert len(set(requested_history)) > 1
    assert result.max_force_jump < 0.05
    assert result.max_alpha_jump < 0.10
    assert result.max_active_patch_count <= config.max_active_patches
    assert result.mean_runtime_ratio < result.full_patch_runtime_ratio
    assert result.mean_projection_error < result.mean_coarse_projection_error


def test_adaptive_patch_activation_writes_phase7_diagnostics(tmp_path) -> None:
    case = _phase7_case()
    result = run_adaptive_patch_sequence(
        case["fem"].mesh,
        case["surface"],
        case["medium"],
        case["medium_bases"],
        _sliding_frames(case["surface"], steps=21),
        penalty=100.0,
        config=AdaptivePatchActivationConfig(neighbor_depth=1, deactivate_delay=4, alpha_blend=0.3, max_active_patches=8),
        coarse_patch_level=case["coarse"],
        coarse_load_bases=case["coarse_bases"],
    )

    write_adaptive_patch_tables(result, tmp_path)

    assert (tmp_path / "active_patch_history.csv").read_text(encoding="utf-8").startswith("step,requested_patch_ids")
    assert "force_jump" in (tmp_path / "force_jump.csv").read_text(encoding="utf-8")
    assert "alpha_jump" in (tmp_path / "alpha_jump.csv").read_text(encoding="utf-8")
    assert "mean_active_runtime_ratio" in (tmp_path / "runtime_vs_active_patch.csv").read_text(encoding="utf-8")
    assert "Adaptive Patch Summary" in (tmp_path / "adaptive_patch_summary.md").read_text(encoding="utf-8")


def _phase7_case():
    fem = cantilever_block(nx=2, ny=8, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(4, 2))
    coarse = hierarchy.level("coarse")
    medium = hierarchy.level("medium")
    return {
        "fem": fem,
        "surface": surface,
        "coarse": coarse,
        "medium": medium,
        "coarse_bases": build_normal_patch_load_bases(fem.mesh, surface, coarse.patches),
        "medium_bases": build_normal_patch_load_bases(fem.mesh, surface, medium.patches),
    }


def _right_side_patches(patches):
    right = [patch for patch in patches if patch.key[0] == 0 and patch.key[1] == 1]
    assert right
    return sorted(right, key=lambda patch: (patch.centroid[1], patch.centroid[2]))


def _single_patch_contact_frame(surface, patch, penetration: float) -> tuple[np.ndarray, np.ndarray]:
    points = []
    areas = []
    for triangle_id in patch.triangle_indices:
        points.append(surface.centroids[triangle_id] - penetration * surface.normals[triangle_id])
        areas.append(surface.areas[triangle_id])
    return np.asarray(points, dtype=float), np.asarray(areas, dtype=float)


def _sliding_frames(surface, steps: int) -> list[tuple[np.ndarray, np.ndarray]]:
    plus_triangles = [tri_id for tri_id, normal in enumerate(surface.normals) if normal[0] > 0.9]
    yz = surface.centroids[plus_triangles][:, 1:3]
    y_min, y_max = float(yz[:, 0].min()), float(yz[:, 0].max())
    centers = np.linspace(y_min + 0.15, y_max - 0.15, steps)
    radius = 0.38
    base_penetration = 0.01
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for y0 in centers:
        points = []
        areas = []
        for triangle_id in plus_triangles:
            centroid = surface.centroids[triangle_id]
            r2 = float((centroid[1] - y0) ** 2 + centroid[2] ** 2)
            weight = max(0.0, 1.0 - r2 / (radius * radius))
            if weight <= 0.0:
                continue
            penetration = base_penetration * weight
            points.append(centroid - penetration * surface.normals[triangle_id])
            areas.append(surface.areas[triangle_id])
        frames.append((np.asarray(points, dtype=float), np.asarray(areas, dtype=float)))
    return frames
