from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.modal_ilc import (
    ReducedState,
    build_normal_patch_load_bases,
    build_patch_residual_basis,
    solve_online_residual_contact_step,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def main() -> None:
    fem = cantilever_block(nx=3, ny=2, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, level.patches)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=3)
    residual_basis = build_patch_residual_basis(fem.K, load_bases, low.modes, free_dofs=fem.free_dofs)
    right_patch = sorted(
        [patch for patch in level.patches if patch.key[0] == 0 and patch.key[1] == 1],
        key=lambda patch: (patch.centroid[1], patch.centroid[2]),
    )[0]
    points, areas = _distributed_patch_contact_points(surface, [right_patch], penetration=0.005)
    state = ReducedState(r=np.zeros(3), theta=np.zeros(3), eta_k=np.zeros(low.modes.shape[1]))

    result = solve_online_residual_contact_step(
        fem.mesh,
        surface,
        level,
        low.modes,
        state,
        {block.patch_id: block for block in residual_basis.blocks},
        load_bases,
        points,
        penalty=20.0,
        sample_areas=areas,
        max_iterations=16,
        tolerance=1.0e-10,
    )
    output_dir = Path("outputs/online_residual_coupling")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alpha_iteration.csv").write_text(_iteration_csv(result), encoding="utf-8")
    (output_dir / "online_residual_summary.md").write_text(_summary(result), encoding="utf-8")

    print(f"converged: {result.converged}")
    print(f"iterations: {len(result.iterations)}")
    print(f"dynamic state dimension: {result.dynamic_state_dimension}")
    print(f"active patches: {result.active_set.active_patch_ids}")
    print(f"final alpha residual: {result.final_alpha_residual:.6e}")
    print(f"alpha residual reduction: {result.alpha_residual_reduction:.6e}")
    print(f"normal force: {result.normal_force:.6e}")
    print(f"wrote {output_dir / 'alpha_iteration.csv'}")
    print(f"wrote {output_dir / 'online_residual_summary.md'}")


def _distributed_patch_contact_points(surface, patches, penetration: float) -> tuple[np.ndarray, np.ndarray]:
    points = []
    areas = []
    for patch in patches:
        for triangle_id in patch.triangle_indices:
            points.append(surface.centroids[triangle_id] - penetration * surface.normals[triangle_id])
            areas.append(surface.areas[triangle_id])
    return np.asarray(points, dtype=float), np.asarray(areas, dtype=float)


def _iteration_csv(result) -> str:
    lines = ["iteration,active_patch_count,alpha_update_norm,min_gap,normal_force,residual_deformation_norm\n"]
    for item in result.iterations:
        lines.append(
            f"{item.iteration},{len(item.active_patch_ids)},{item.alpha_update_norm:.17g},"
            f"{item.min_gap:.17g},{item.normal_force:.17g},{item.residual_deformation_norm:.17g}\n"
        )
    return "".join(lines)


def _summary(result) -> str:
    return (
        "# Online Residual Coupling Summary\n\n"
        f"Converged: {result.converged}\n"
        f"Iterations: {len(result.iterations)}\n"
        f"Dynamic state dimension: {result.dynamic_state_dimension}\n"
        f"Active patches: {result.active_set.active_patch_ids}\n"
        f"Final alpha residual: {result.final_alpha_residual:.6g}\n"
        f"Alpha residual reduction: {result.alpha_residual_reduction:.6g}\n"
        f"Normal force: {result.normal_force:.6g}\n"
    )


if __name__ == "__main__":
    main()
