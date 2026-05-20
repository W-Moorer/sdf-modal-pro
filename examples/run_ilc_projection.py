from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_ilc import (
    build_normal_patch_load_bases,
    detect_sdf_contact_samples,
    project_contact_samples_to_patch_ilc,
    write_ilc_projection_tables,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def main() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    active_patches = _right_side_patches(level.patches)[:2]
    points, areas = _distributed_patch_contact_points(surface, active_patches, penetration=0.015)
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, level.patches)
    samples = detect_sdf_contact_samples(points, surface, level, penalty=200.0, sample_areas=areas)
    projection = project_contact_samples_to_patch_ilc(fem.mesh, surface, samples, load_bases)
    output_dir = Path("outputs/ilc_projection")
    write_ilc_projection_tables(projection, output_dir)

    print(f"contact samples: {samples.sample_count}")
    print(f"active patches: {projection.active_patch_count} / {len(level.patches)}")
    print(f"total force error: {projection.total_force_error:.6e}")
    print(f"total normal force error: {projection.total_normal_force_error:.6e}")
    print(f"total moment error: {projection.total_moment_error:.6e}")
    print(f"wrote {output_dir / 'contact_to_patch_map.csv'}")
    print(f"wrote {output_dir / 'force_projection_error.csv'}")
    print(f"wrote {output_dir / 'moment_projection_error.csv'}")
    print(f"wrote {output_dir / 'ilc_projection_summary.md'}")


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
    return sorted(right, key=lambda patch: (patch.centroid[1], patch.centroid[2]))


if __name__ == "__main__":
    main()
