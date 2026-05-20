from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.modal_ilc.ilc_projection import ActivePatchSet
from modal_contact_rom.modal_ilc.patch_load_basis import PatchLoadBasis, assemble_load_basis
from modal_contact_rom.sdf_query.mesh_distance import SignedDistanceResult, query_signed_distances
from modal_contact_rom.surface_patch.extract import SurfaceMesh
from modal_contact_rom.surface_patch.patch_hierarchy import PatchLevel


@dataclass(frozen=True)
class ContactSampleSet:
    points: np.ndarray
    triangle_indices: np.ndarray
    patch_ids: np.ndarray
    gaps: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    normal_forces: np.ndarray
    forces: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        triangle_indices = np.asarray(self.triangle_indices, dtype=np.int64)
        patch_ids = np.asarray(self.patch_ids, dtype=np.int64)
        gaps = np.asarray(self.gaps, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        areas = np.asarray(self.areas, dtype=float)
        normal_forces = np.asarray(self.normal_forces, dtype=float)
        forces = np.asarray(self.forces, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n_samples, 3)")
        n_samples = points.shape[0]
        if triangle_indices.shape != (n_samples,) or patch_ids.shape != (n_samples,):
            raise ValueError("triangle_indices and patch_ids must have one entry per sample")
        if gaps.shape != (n_samples,) or areas.shape != (n_samples,) or normal_forces.shape != (n_samples,):
            raise ValueError("gaps, areas, and normal_forces must have one entry per sample")
        if normals.shape != (n_samples, 3) or forces.shape != (n_samples, 3):
            raise ValueError("normals and forces must have shape (n_samples, 3)")
        if np.any(areas < 0.0):
            raise ValueError("sample areas must be non-negative")
        if np.any(normal_forces < -1.0e-14):
            raise ValueError("normal forces must be non-negative")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "triangle_indices", triangle_indices)
        object.__setattr__(self, "patch_ids", patch_ids)
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "areas", areas)
        object.__setattr__(self, "normal_forces", normal_forces)
        object.__setattr__(self, "forces", forces)

    @classmethod
    def empty(cls) -> ContactSampleSet:
        return cls(
            points=np.zeros((0, 3), dtype=float),
            triangle_indices=np.zeros(0, dtype=np.int64),
            patch_ids=np.zeros(0, dtype=np.int64),
            gaps=np.zeros(0, dtype=float),
            normals=np.zeros((0, 3), dtype=float),
            areas=np.zeros(0, dtype=float),
            normal_forces=np.zeros(0, dtype=float),
            forces=np.zeros((0, 3), dtype=float),
        )

    @property
    def sample_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def active_patch_ids(self) -> tuple[int, ...]:
        return tuple(int(patch_id) for patch_id in np.unique(self.patch_ids))

    @property
    def total_force(self) -> np.ndarray:
        return np.sum(self.forces, axis=0) if self.sample_count else np.zeros(3, dtype=float)

    @property
    def total_normal_force(self) -> float:
        return float(np.sum(self.normal_forces))

    def total_moment(self, origin: np.ndarray | None = None) -> np.ndarray:
        if origin is None:
            origin = np.zeros(3, dtype=float)
        origin = np.asarray(origin, dtype=float)
        return np.sum(np.cross(self.points - origin[None, :], self.forces), axis=0)


@dataclass(frozen=True)
class PatchILCProjection:
    samples: ContactSampleSet
    active_set: ActivePatchSet
    active_load_bases: tuple[PatchLoadBasis, ...]
    projected_force_vector: np.ndarray
    sample_lumped_force_vector: np.ndarray
    total_force_error: float
    total_normal_force_error: float
    total_moment_error: float

    @property
    def active_patch_count(self) -> int:
        return len(self.active_set.active_patch_ids)


def detect_sdf_contact_samples(
    points: np.ndarray,
    surface: SurfaceMesh,
    patch_level: PatchLevel,
    penalty: float,
    sample_areas: np.ndarray | None = None,
    contact_offset: float = 0.0,
) -> ContactSampleSet:
    """Query SDF contact samples and map active samples to patch ids."""

    query = query_signed_distances(points, surface)
    return contact_samples_from_sdf_query(
        points,
        surface,
        patch_level,
        penalty,
        query,
        sample_areas=sample_areas,
        contact_offset=contact_offset,
    )


def contact_samples_from_sdf_query(
    points: np.ndarray,
    surface: SurfaceMesh,
    patch_level: PatchLevel,
    penalty: float,
    query: SignedDistanceResult,
    sample_areas: np.ndarray | None = None,
    contact_offset: float = 0.0,
) -> ContactSampleSet:
    """Map a cached SDF query to contact samples for one patch level."""

    if penalty < 0.0:
        raise ValueError("penalty must be non-negative")
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim == 1:
        point_array = point_array[None, :]
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if query.distances.shape != (point_array.shape[0],):
        raise ValueError("cached SDF query must contain one distance per query point")
    if query.closest_points.shape != point_array.shape or query.triangle_indices.shape != (point_array.shape[0],):
        raise ValueError("cached SDF query shape does not match query points")

    distances = query.distances
    penetration = np.maximum(contact_offset - distances, 0.0)
    active = penetration > 0.0
    if not np.any(active):
        return ContactSampleSet.empty()

    areas = _sample_area_vector(sample_areas, point_array, surface)
    active_triangles = query.triangle_indices[active]
    active_areas = areas[active]
    normals = surface.normals[active_triangles]
    normal_forces = penalty * penetration[active] * active_areas
    forces = normals * normal_forces[:, None]
    patch_ids = patch_level.primary_patch_ids[active_triangles]
    return ContactSampleSet(
        points=query.closest_points[active],
        triangle_indices=active_triangles,
        patch_ids=patch_ids,
        gaps=distances[active] - contact_offset,
        normals=normals,
        areas=active_areas,
        normal_forces=normal_forces,
        forces=forces,
    )


def project_contact_samples_to_patch_ilc(
    mesh: Mesh,
    surface: SurfaceMesh,
    samples: ContactSampleSet,
    load_bases: Sequence[PatchLoadBasis],
    origin: np.ndarray | None = None,
) -> PatchILCProjection:
    """Aggregate SDF contact samples into active patch ILC amplitudes."""

    sample_lumped = assemble_sample_lumped_force_vector(mesh, surface, samples)
    if samples.sample_count == 0:
        zeros = np.zeros(mesh.n_dofs, dtype=float)
        return PatchILCProjection(
            samples=samples,
            active_set=ActivePatchSet.empty(),
            active_load_bases=(),
            projected_force_vector=zeros,
            sample_lumped_force_vector=zeros,
            total_force_error=0.0,
            total_normal_force_error=0.0,
            total_moment_error=0.0,
        )

    basis_by_patch = {basis.patch_id: basis for basis in load_bases}
    index_by_patch = {basis.patch_id: index for index, basis in enumerate(load_bases)}
    active_patch_ids = tuple(patch_id for patch_id in samples.active_patch_ids if patch_id in basis_by_patch)
    active_bases = tuple(basis_by_patch[patch_id] for patch_id in active_patch_ids)
    alpha = np.asarray([_normal_resultant_for_patch(samples, basis) for basis in active_bases], dtype=float)
    alpha = np.maximum(alpha, 0.0)
    active_set = ActivePatchSet(
        active_patch_ids=active_patch_ids,
        active_load_basis_indices=tuple(index_by_patch[patch_id] for patch_id in active_patch_ids),
        alpha=alpha,
    )
    projected = assemble_load_basis(active_bases) @ alpha if active_bases else np.zeros(mesh.n_dofs, dtype=float)
    projected_total_force = total_nodal_force(projected)
    sample_total_force = samples.total_force
    sample_normal_force = samples.total_normal_force
    projected_normal_force = float(np.sum(alpha))
    force_error = _relative_norm(projected_total_force - sample_total_force, sample_total_force)
    normal_error = abs(projected_normal_force - sample_normal_force) / max(abs(sample_normal_force), 1.0e-30)
    moment_error = _relative_norm(
        total_nodal_moment(mesh, projected, origin),
        samples.total_moment(origin),
        subtract=True,
    )
    return PatchILCProjection(
        samples=samples,
        active_set=active_set,
        active_load_bases=active_bases,
        projected_force_vector=projected,
        sample_lumped_force_vector=sample_lumped,
        total_force_error=force_error,
        total_normal_force_error=normal_error,
        total_moment_error=moment_error,
    )


def assemble_sample_lumped_force_vector(mesh: Mesh, surface: SurfaceMesh, samples: ContactSampleSet) -> np.ndarray:
    """Distribute each contact sample force equally to its closest triangle nodes."""

    force = np.zeros(mesh.n_dofs, dtype=float)
    if samples.sample_count == 0:
        return force
    nodal_force = force.reshape(mesh.n_nodes, 3)
    triangle_nodes = surface.triangles[samples.triangle_indices].reshape(-1)
    distributed_forces = np.repeat(samples.forces / 3.0, 3, axis=0)
    np.add.at(nodal_force, triangle_nodes, distributed_forces)
    return force


def total_nodal_force(force_vector: np.ndarray) -> np.ndarray:
    force = np.asarray(force_vector, dtype=float)
    if force.ndim != 1 or force.size % 3 != 0:
        raise ValueError("force_vector must be a flat vector with 3 DOFs per node")
    return np.sum(force.reshape(-1, 3), axis=0)


def total_nodal_moment(mesh: Mesh, force_vector: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
    if origin is None:
        origin = np.zeros(3, dtype=float)
    origin = np.asarray(origin, dtype=float)
    force = np.asarray(force_vector, dtype=float).reshape(mesh.n_nodes, 3)
    return np.sum(np.cross(mesh.nodes - origin[None, :], force), axis=0)


def write_ilc_projection_tables(projection: PatchILCProjection, output_dir: str | Path) -> None:
    """Write Phase-5 contact-to-patch ILC diagnostics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    map_lines = ["sample_id,triangle_id,patch_id,gap,area,normal_force,fx,fy,fz\n"]
    for sample_id in range(projection.samples.sample_count):
        force = projection.samples.forces[sample_id]
        map_lines.append(
            f"{sample_id},{int(projection.samples.triangle_indices[sample_id])},"
            f"{int(projection.samples.patch_ids[sample_id])},{projection.samples.gaps[sample_id]:.17g},"
            f"{projection.samples.areas[sample_id]:.17g},{projection.samples.normal_forces[sample_id]:.17g},"
            f"{force[0]:.17g},{force[1]:.17g},{force[2]:.17g}\n"
        )

    force_lines = [
        "active_patch_count,total_force_error,total_normal_force_error,lumped_vector_relative_error\n",
        f"{projection.active_patch_count},{projection.total_force_error:.17g},"
        f"{projection.total_normal_force_error:.17g},"
        f"{_relative_norm(projection.projected_force_vector - projection.sample_lumped_force_vector, projection.sample_lumped_force_vector):.17g}\n",
    ]
    moment_lines = [
        "active_patch_count,total_moment_error\n",
        f"{projection.active_patch_count},{projection.total_moment_error:.17g}\n",
    ]
    summary = (
        "# ILC Projection Summary\n\n"
        f"Contact samples: {projection.samples.sample_count}\n"
        f"Active patches: {projection.active_patch_count}\n"
        f"Total force error: {projection.total_force_error:.6g}\n"
        f"Total normal force error: {projection.total_normal_force_error:.6g}\n"
        f"Total moment error: {projection.total_moment_error:.6g}\n"
    )
    (output_path / "contact_to_patch_map.csv").write_text("".join(map_lines), encoding="utf-8")
    (output_path / "force_projection_error.csv").write_text("".join(force_lines), encoding="utf-8")
    (output_path / "moment_projection_error.csv").write_text("".join(moment_lines), encoding="utf-8")
    (output_path / "ilc_projection_summary.md").write_text(summary, encoding="utf-8")


def _normal_resultant_for_patch(samples: ContactSampleSet, basis: PatchLoadBasis) -> float:
    if basis.resultant is None:
        direction = total_nodal_force(basis.B_j[:, 0])
    else:
        direction = basis.resultant
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError(f"patch load basis {basis.patch_id} has zero resultant")
    direction = direction / norm
    mask = samples.patch_ids == basis.patch_id
    return float(np.sum(samples.forces[mask] @ direction))


def _sample_area_vector(sample_areas: np.ndarray | None, points: np.ndarray, surface: SurfaceMesh) -> np.ndarray:
    point_array = np.asarray(points, dtype=float)
    point_count = 1 if point_array.ndim == 1 else int(point_array.shape[0])
    if sample_areas is None:
        return np.ones(point_count, dtype=float)
    areas = np.asarray(sample_areas, dtype=float)
    if areas.shape != (point_count,):
        raise ValueError("sample_areas must have one entry per query point")
    if np.any(areas < 0.0):
        raise ValueError("sample_areas must be non-negative")
    return areas


def _relative_norm(error_or_value: np.ndarray, reference: np.ndarray, subtract: bool = False) -> float:
    if subtract:
        error = np.asarray(error_or_value, dtype=float) - np.asarray(reference, dtype=float)
    else:
        error = np.asarray(error_or_value, dtype=float)
    denom = max(float(np.linalg.norm(reference)), 1.0e-30)
    return float(np.linalg.norm(error) / denom)
