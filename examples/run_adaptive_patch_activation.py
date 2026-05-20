from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_ilc import (
    AdaptivePatchActivationConfig,
    build_normal_patch_load_bases,
    run_adaptive_patch_sequence,
    write_adaptive_patch_tables,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def main() -> None:
    fem = cantilever_block(nx=2, ny=8, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(4, 2))
    coarse = hierarchy.level("coarse")
    medium = hierarchy.level("medium")
    coarse_bases = build_normal_patch_load_bases(fem.mesh, surface, coarse.patches)
    medium_bases = build_normal_patch_load_bases(fem.mesh, surface, medium.patches)
    config = AdaptivePatchActivationConfig(
        neighbor_depth=1,
        deactivate_delay=4,
        alpha_blend=0.3,
        max_active_patches=8,
    )
    result = run_adaptive_patch_sequence(
        fem.mesh,
        surface,
        medium,
        medium_bases,
        _sliding_frames(surface, steps=41),
        penalty=100.0,
        config=config,
        coarse_patch_level=coarse,
        coarse_load_bases=coarse_bases,
    )
    output_dir = Path("outputs/adaptive_patch")
    write_adaptive_patch_tables(result, output_dir)

    print(f"max force jump: {result.max_force_jump:.6e}")
    print(f"max alpha jump: {result.max_alpha_jump:.6e}")
    print(f"max active patches: {result.max_active_patch_count} / {result.total_patch_count}")
    print(f"mean runtime ratio: {result.mean_runtime_ratio:.6e}")
    print(f"mean adaptive projection error: {result.mean_projection_error:.6e}")
    print(f"mean coarse-only projection error: {result.mean_coarse_projection_error:.6e}")
    print(f"wrote {output_dir / 'active_patch_history.csv'}")
    print(f"wrote {output_dir / 'force_jump.csv'}")
    print(f"wrote {output_dir / 'alpha_jump.csv'}")
    print(f"wrote {output_dir / 'runtime_vs_active_patch.csv'}")
    print(f"wrote {output_dir / 'adaptive_patch_summary.md'}")


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
            points.append(centroid - base_penetration * weight * surface.normals[triangle_id])
            areas.append(surface.areas[triangle_id])
        frames.append((np.asarray(points, dtype=float), np.asarray(areas, dtype=float)))
    return frames


if __name__ == "__main__":
    main()
