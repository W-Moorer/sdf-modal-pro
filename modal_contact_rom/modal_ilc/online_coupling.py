from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.modal_ilc.ilc_projection import ActivePatchSet
from modal_contact_rom.modal_ilc.patch_load_basis import PatchLoadBasis
from modal_contact_rom.modal_ilc.patch_residual_modes import PatchResidualBlock
from modal_contact_rom.modal_ilc.reduced_state import ReducedState
from modal_contact_rom.modal_ilc.residual_recovery import recover_residual_deformation
from modal_contact_rom.modal_ilc.sdf_projection import (
    ContactSampleSet,
    PatchILCProjection,
    detect_sdf_contact_samples,
    project_contact_samples_to_patch_ilc,
)
from modal_contact_rom.surface_patch.extract import SurfaceMesh
from modal_contact_rom.surface_patch.patch_hierarchy import PatchLevel


@dataclass(frozen=True)
class OnlineResidualIteration:
    iteration: int
    active_patch_ids: tuple[int, ...]
    alpha: np.ndarray
    alpha_update_norm: float
    min_gap: float
    normal_force: float
    residual_deformation_norm: float


@dataclass(frozen=True)
class OnlineResidualStepResult:
    state: ReducedState
    active_set: ActivePatchSet
    modal_displacement: np.ndarray
    residual_deformation: np.ndarray
    displacement: np.ndarray
    samples: ContactSampleSet
    projection: PatchILCProjection
    iterations: tuple[OnlineResidualIteration, ...]
    converged: bool

    @property
    def dynamic_state_dimension(self) -> int:
        return self.state.dynamic_dimension

    @property
    def final_alpha_residual(self) -> float:
        if not self.iterations:
            return 0.0
        return float(self.iterations[-1].alpha_update_norm)

    @property
    def alpha_residual_reduction(self) -> float:
        if len(self.iterations) < 2:
            return np.inf
        initial = max(float(self.iterations[0].alpha_update_norm), 1.0e-30)
        final = max(float(self.iterations[-1].alpha_update_norm), 1.0e-30)
        return initial / final

    @property
    def normal_force(self) -> float:
        return float(self.projection.samples.total_normal_force)


def solve_online_residual_contact_step(
    mesh: Mesh,
    surface: SurfaceMesh,
    patch_level: PatchLevel,
    retained_modes: np.ndarray,
    state: ReducedState,
    residual_blocks: Mapping[int, PatchResidualBlock] | Sequence[PatchResidualBlock],
    load_bases: Sequence[PatchLoadBasis],
    contact_points: np.ndarray,
    penalty: float,
    sample_areas: np.ndarray | None = None,
    previous_active_set: ActivePatchSet | None = None,
    contact_offset: float = 0.0,
    max_iterations: int = 25,
    tolerance: float = 1.0e-8,
    relaxation: float = 1.0,
    include_residual: bool = True,
) -> OnlineResidualStepResult:
    """Run one frozen-tangent/fixed-point contact-ILC update."""

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1]")

    Psi = np.asarray(retained_modes, dtype=float)
    if Psi.ndim != 2 or Psi.shape[0] != mesh.n_dofs:
        raise ValueError("retained_modes must have shape (mesh.n_dofs, n_modes)")
    if Psi.shape[1] != state.eta_k.size:
        raise ValueError("state eta_k length must match retained_modes columns")

    modal_displacement = Psi @ state.eta_k
    active_set = ActivePatchSet.empty() if previous_active_set is None else previous_active_set
    iterations: list[OnlineResidualIteration] = []
    projection: PatchILCProjection | None = None
    samples: ContactSampleSet = ContactSampleSet.empty()
    residual_deformation = np.zeros(mesh.n_dofs, dtype=float)
    displacement = modal_displacement.copy()
    converged = False

    for iteration in range(max_iterations):
        residual_deformation = _residual_deformation(residual_blocks, active_set, mesh.n_dofs, include_residual)
        displacement = modal_displacement + residual_deformation
        displaced_surface = surface_with_displacement(surface, displacement)
        samples = detect_sdf_contact_samples(
            contact_points,
            displaced_surface,
            patch_level,
            penalty=penalty,
            sample_areas=sample_areas,
            contact_offset=contact_offset,
        )
        projection = project_contact_samples_to_patch_ilc(mesh, displaced_surface, samples, load_bases)
        next_active_set = _relaxed_active_set(active_set, projection.active_set, relaxation)
        update_norm = _active_set_delta_norm(active_set, projection.active_set)
        iterations.append(
            OnlineResidualIteration(
                iteration=iteration,
                active_patch_ids=next_active_set.active_patch_ids,
                alpha=next_active_set.alpha.copy(),
                alpha_update_norm=update_norm,
                min_gap=_min_gap(samples),
                normal_force=samples.total_normal_force,
                residual_deformation_norm=float(np.linalg.norm(residual_deformation)),
            )
        )
        active_set = next_active_set
        if update_norm <= tolerance:
            converged = True
            break

    residual_deformation = _residual_deformation(residual_blocks, active_set, mesh.n_dofs, include_residual)
    displacement = modal_displacement + residual_deformation
    displaced_surface = surface_with_displacement(surface, displacement)
    samples = detect_sdf_contact_samples(
        contact_points,
        displaced_surface,
        patch_level,
        penalty=penalty,
        sample_areas=sample_areas,
        contact_offset=contact_offset,
    )
    projection = project_contact_samples_to_patch_ilc(mesh, displaced_surface, samples, load_bases)
    final_update = _active_set_delta_norm(active_set, projection.active_set)
    if iterations and final_update < iterations[-1].alpha_update_norm:
        iterations.append(
            OnlineResidualIteration(
                iteration=len(iterations),
                active_patch_ids=projection.active_set.active_patch_ids,
                alpha=projection.active_set.alpha.copy(),
                alpha_update_norm=final_update,
                min_gap=_min_gap(samples),
                normal_force=samples.total_normal_force,
                residual_deformation_norm=float(np.linalg.norm(residual_deformation)),
            )
        )
        active_set = projection.active_set
        converged = converged or final_update <= tolerance

    return OnlineResidualStepResult(
        state=state,
        active_set=active_set,
        modal_displacement=modal_displacement,
        residual_deformation=residual_deformation,
        displacement=displacement,
        samples=samples,
        projection=projection,
        iterations=tuple(iterations),
        converged=converged,
    )


def surface_with_displacement(surface: SurfaceMesh, displacement: np.ndarray) -> SurfaceMesh:
    """Return a deformed surface mesh used by online gap/contact queries."""

    displacement_array = np.asarray(displacement, dtype=float)
    if displacement_array.shape != (surface.nodes.shape[0] * 3,):
        raise ValueError("displacement must have one 3D vector per surface node")
    nodes = surface.nodes + displacement_array.reshape((-1, 3))
    areas, normals, centroids = _triangle_geometry(nodes, surface.triangles)
    return SurfaceMesh(nodes=nodes, triangles=surface.triangles, areas=areas, normals=normals, centroids=centroids)


def _residual_deformation(
    residual_blocks: Mapping[int, PatchResidualBlock] | Sequence[PatchResidualBlock],
    active_set: ActivePatchSet,
    n_dofs: int,
    include_residual: bool,
) -> np.ndarray:
    if not include_residual or active_set.is_empty:
        return np.zeros(n_dofs, dtype=float)
    return recover_residual_deformation(residual_blocks, active_set, n_dofs=n_dofs)


def _relaxed_active_set(
    previous: ActivePatchSet,
    projected: ActivePatchSet,
    relaxation: float,
) -> ActivePatchSet:
    if projected.is_empty:
        return ActivePatchSet.empty()
    previous_by_patch = _alpha_by_patch(previous)
    relaxed = np.empty_like(projected.alpha)
    for index, patch_id in enumerate(projected.active_patch_ids):
        old_value = previous_by_patch.get(int(patch_id), 0.0)
        relaxed[index] = (1.0 - relaxation) * old_value + relaxation * projected.alpha[index]
    return ActivePatchSet(
        active_patch_ids=projected.active_patch_ids,
        active_load_basis_indices=projected.active_load_basis_indices,
        alpha=relaxed,
    )


def _active_set_delta_norm(previous: ActivePatchSet, projected: ActivePatchSet) -> float:
    if projected.is_empty and previous.is_empty:
        return 0.0
    previous_by_patch = _alpha_by_patch(previous)
    all_patch_ids = tuple(sorted(set(previous.active_patch_ids) | set(projected.active_patch_ids)))
    if not all_patch_ids:
        return 0.0
    projected_by_patch = _alpha_by_patch(projected)
    delta = np.asarray(
        [projected_by_patch.get(patch_id, 0.0) - previous_by_patch.get(patch_id, 0.0) for patch_id in all_patch_ids],
        dtype=float,
    )
    reference = np.asarray([projected_by_patch.get(patch_id, 0.0) for patch_id in all_patch_ids], dtype=float)
    return float(np.linalg.norm(delta) / max(1.0, np.linalg.norm(reference)))


def _alpha_by_patch(active_set: ActivePatchSet) -> dict[int, float]:
    return {int(patch_id): float(active_set.alpha[index]) for index, patch_id in enumerate(active_set.active_patch_ids)}


def _min_gap(samples: ContactSampleSet) -> float:
    if samples.sample_count == 0:
        return 0.0
    return float(np.min(samples.gaps))


def _triangle_geometry(nodes: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pa = nodes[triangles[:, 0]]
    pb = nodes[triangles[:, 1]]
    pc = nodes[triangles[:, 2]]
    raw_normals = np.cross(pb - pa, pc - pa)
    raw_lengths = np.linalg.norm(raw_normals, axis=1)
    areas = 0.5 * raw_lengths
    normals = np.zeros_like(raw_normals)
    valid = raw_lengths > 0.0
    normals[valid] = raw_normals[valid] / raw_lengths[valid, None]
    centroids = (pa + pb + pc) / 3.0
    return areas, normals, centroids
