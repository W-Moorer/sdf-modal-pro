from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modal_contact_rom.surface_patch.extract import SurfaceMesh, SurfacePatch, partition_surface_patches


@dataclass(frozen=True)
class PatchLevel:
    name: str
    patches: tuple[SurfacePatch, ...]
    triangle_weights: np.ndarray
    primary_patch_ids: np.ndarray
    adjacency: dict[int, tuple[int, ...]]
    overlap: bool = False

    def __post_init__(self) -> None:
        weights = np.asarray(self.triangle_weights, dtype=float)
        primary = np.asarray(self.primary_patch_ids, dtype=np.int64)
        if weights.ndim != 2:
            raise ValueError("triangle_weights must be a two-dimensional matrix")
        if weights.shape[1] != len(self.patches):
            raise ValueError("triangle_weights columns must match patch count")
        if primary.ndim != 1 or primary.size != weights.shape[0]:
            raise ValueError("primary_patch_ids must have one entry per surface triangle")
        object.__setattr__(self, "triangle_weights", weights)
        object.__setattr__(self, "primary_patch_ids", primary)

    @property
    def patch_count(self) -> int:
        return len(self.patches)

    @property
    def triangle_count(self) -> int:
        return int(self.triangle_weights.shape[0])

    def coverage_ratio(self) -> float:
        if self.triangle_count == 0:
            return 1.0
        return float(np.count_nonzero(self.triangle_weights.sum(axis=1) > 0.0) / self.triangle_count)

    def weighted_patch_areas(self, surface: SurfaceMesh) -> np.ndarray:
        if surface.areas.size != self.triangle_count:
            raise ValueError("surface triangle count must match this patch level")
        return self.triangle_weights.T @ surface.areas

    def covered_area(self, surface: SurfaceMesh) -> float:
        return float(np.sum(self.weighted_patch_areas(surface)))

    def surface_area_error(self, surface: SurfaceMesh) -> float:
        return abs(self.covered_area(surface) - float(np.sum(surface.areas)))

    def patch_by_id(self, patch_id: int) -> SurfacePatch:
        for patch in self.patches:
            if patch.patch_id == patch_id:
                return patch
        raise KeyError(f"unknown patch_id {patch_id}")


@dataclass(frozen=True)
class PatchHierarchy:
    surface: SurfaceMesh
    levels: tuple[PatchLevel, ...]

    def level(self, name: str) -> PatchLevel:
        for level in self.levels:
            if level.name == name:
                return level
        raise KeyError(f"unknown patch hierarchy level {name!r}")


def build_patch_hierarchy(
    surface: SurfaceMesh,
    coarse_bins: tuple[int, int] = (1, 1),
    medium_bins: tuple[int, int] = (2, 2),
    fine_bins: tuple[int, int] | None = None,
    include_overlap_medium: bool = True,
    include_overlap_fine: bool = True,
) -> PatchHierarchy:
    """Build the Phase-2 coarse/medium/fine hierarchy over the full exterior surface."""

    coarse = build_patch_level(surface, "coarse", partition_surface_patches(surface, coarse_bins), overlap=False)
    medium_patches = partition_surface_patches(surface, medium_bins)
    levels = [coarse, build_patch_level(surface, "medium", medium_patches, overlap=False)]
    if include_overlap_medium:
        levels.append(build_patch_level(surface, "medium_overlap", medium_patches, overlap=True))
    if fine_bins is not None:
        fine_patches = partition_surface_patches(surface, fine_bins)
        levels.append(build_patch_level(surface, "fine", fine_patches, overlap=False))
        if include_overlap_fine:
            levels.append(build_patch_level(surface, "fine_overlap", fine_patches, overlap=True))
    return PatchHierarchy(surface=surface, levels=tuple(levels))


def build_patch_level(
    surface: SurfaceMesh,
    name: str,
    patches: list[SurfacePatch] | tuple[SurfacePatch, ...],
    overlap: bool = False,
) -> PatchLevel:
    """Build assignment weights and edge adjacency for one patch level."""

    patch_tuple = tuple(patches)
    primary_patch_ids, patch_id_to_column = _primary_patch_ids(surface, patch_tuple)
    adjacency = _patch_adjacency(surface, primary_patch_ids)
    weights = _triangle_weights(primary_patch_ids, patch_id_to_column, adjacency, overlap)
    return PatchLevel(
        name=name,
        patches=patch_tuple,
        triangle_weights=weights,
        primary_patch_ids=primary_patch_ids,
        adjacency=adjacency,
        overlap=overlap,
    )


def write_patch_hierarchy_tables(hierarchy: PatchHierarchy, output_dir: str | Path) -> None:
    """Write compact CSV diagnostics for coverage, area, and adjacency."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats_lines = ["level,patch_id,area,weighted_area,neighbor_count,triangle_count\n"]
    coverage_lines = ["level,coverage_ratio,surface_area,covered_area,area_error,overlap\n"]
    for level in hierarchy.levels:
        weighted_areas = level.weighted_patch_areas(hierarchy.surface)
        for column, patch in enumerate(level.patches):
            stats_lines.append(
                f"{level.name},{patch.patch_id},{patch.area:.17g},{weighted_areas[column]:.17g},"
                f"{len(level.adjacency.get(patch.patch_id, ()))},{patch.triangle_indices.size}\n"
            )
        surface_area = float(np.sum(hierarchy.surface.areas))
        coverage_lines.append(
            f"{level.name},{level.coverage_ratio():.17g},{surface_area:.17g},"
            f"{level.covered_area(hierarchy.surface):.17g},{level.surface_area_error(hierarchy.surface):.17g},"
            f"{int(level.overlap)}\n"
        )

    (output_path / "patch_stats.csv").write_text("".join(stats_lines), encoding="utf-8")
    (output_path / "patch_coverage.csv").write_text("".join(coverage_lines), encoding="utf-8")
    (output_path / "patch_summary.md").write_text(_patch_summary(hierarchy), encoding="utf-8")


def _primary_patch_ids(
    surface: SurfaceMesh,
    patches: tuple[SurfacePatch, ...],
) -> tuple[np.ndarray, dict[int, int]]:
    primary_patch_ids = np.full(surface.triangles.shape[0], fill_value=-1, dtype=np.int64)
    patch_id_to_column: dict[int, int] = {}
    for column, patch in enumerate(patches):
        patch_id_to_column[patch.patch_id] = column
        for triangle_id in patch.triangle_indices:
            if primary_patch_ids[int(triangle_id)] != -1:
                raise ValueError(f"surface triangle {triangle_id} is assigned to multiple primary patches")
            primary_patch_ids[int(triangle_id)] = patch.patch_id
    missing = np.flatnonzero(primary_patch_ids < 0)
    if missing.size:
        raise ValueError(f"{missing.size} surface triangles are not assigned to a patch")
    return primary_patch_ids, patch_id_to_column


def _triangle_weights(
    primary_patch_ids: np.ndarray,
    patch_id_to_column: dict[int, int],
    adjacency: dict[int, tuple[int, ...]],
    overlap: bool,
) -> np.ndarray:
    weights = np.zeros((primary_patch_ids.size, len(patch_id_to_column)), dtype=float)
    for triangle_id, primary_patch_id in enumerate(primary_patch_ids):
        active_patch_ids = [int(primary_patch_id)]
        if overlap:
            active_patch_ids.extend(adjacency.get(int(primary_patch_id), ()))
        active_columns = [patch_id_to_column[patch_id] for patch_id in active_patch_ids if patch_id in patch_id_to_column]
        value = 1.0 / float(len(active_columns))
        weights[triangle_id, active_columns] = value
    return weights


def _patch_adjacency(surface: SurfaceMesh, primary_patch_ids: np.ndarray) -> dict[int, tuple[int, ...]]:
    neighbors: dict[int, set[int]] = {int(patch_id): set() for patch_id in np.unique(primary_patch_ids)}
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for triangle_id, triangle in enumerate(surface.triangles):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_to_triangles.setdefault(edge, []).append(triangle_id)

    for triangle_ids in edge_to_triangles.values():
        if len(triangle_ids) < 2:
            continue
        patch_ids = sorted({int(primary_patch_ids[triangle_id]) for triangle_id in triangle_ids})
        for patch_id in patch_ids:
            neighbors[patch_id].update(other for other in patch_ids if other != patch_id)

    return {patch_id: tuple(sorted(patch_neighbors)) for patch_id, patch_neighbors in neighbors.items()}


def _patch_summary(hierarchy: PatchHierarchy) -> str:
    lines = ["# Patch Hierarchy Summary\n\n"]
    surface_area = float(np.sum(hierarchy.surface.areas))
    lines.append(f"Surface triangles: {hierarchy.surface.triangles.shape[0]}\n")
    lines.append(f"Surface area: {surface_area:.12g}\n\n")
    for level in hierarchy.levels:
        lines.append(
            f"- {level.name}: patches={level.patch_count}, "
            f"coverage={level.coverage_ratio():.6f}, "
            f"area_error={level.surface_area_error(hierarchy.surface):.3e}, "
            f"overlap={level.overlap}\n"
        )
    return "".join(lines)
