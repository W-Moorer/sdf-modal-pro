from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

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


@dataclass(frozen=True)
class OnlineReducedDynamicState:
    state: ReducedState
    eta_velocity: np.ndarray
    eta_acceleration: np.ndarray
    active_set: ActivePatchSet

    def __post_init__(self) -> None:
        velocity = np.asarray(self.eta_velocity, dtype=float)
        acceleration = np.asarray(self.eta_acceleration, dtype=float)
        if velocity.shape != self.state.eta_k.shape:
            raise ValueError("eta_velocity shape must match state eta_k")
        if acceleration.shape != self.state.eta_k.shape:
            raise ValueError("eta_acceleration shape must match state eta_k")
        object.__setattr__(self, "eta_velocity", velocity)
        object.__setattr__(self, "eta_acceleration", acceleration)


@dataclass(frozen=True)
class OnlineReducedDynamicIteration:
    iteration: int
    active_patch_ids: tuple[int, ...]
    alpha_update_norm: float
    eta_update_norm: float
    reduced_residual_norm: float
    min_gap: float
    normal_force: float
    residual_deformation_norm: float


@dataclass(frozen=True)
class OnlineReducedDynamicStepResult:
    previous: OnlineReducedDynamicState
    next: OnlineReducedDynamicState
    modal_displacement: np.ndarray
    residual_deformation: np.ndarray
    displacement: np.ndarray
    samples: ContactSampleSet
    projection: PatchILCProjection
    iterations: tuple[OnlineReducedDynamicIteration, ...]
    converged: bool

    @property
    def dynamic_state_dimension(self) -> int:
        return self.next.state.dynamic_dimension

    @property
    def final_iteration_residual(self) -> float:
        if not self.iterations:
            return 0.0
        last = self.iterations[-1]
        return float(max(last.alpha_update_norm, last.eta_update_norm, last.reduced_residual_norm))

    @property
    def iteration_residual_reduction(self) -> float:
        if len(self.iterations) < 2:
            return np.inf
        first = self.iterations[0]
        last = self.iterations[-1]
        initial = max(first.alpha_update_norm, first.eta_update_norm, first.reduced_residual_norm, 1.0e-30)
        final = max(last.alpha_update_norm, last.eta_update_norm, last.reduced_residual_norm, 1.0e-30)
        return float(initial / final)


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


def solve_online_residual_dynamic_step(
    K: sp.spmatrix,
    M: sp.spmatrix,
    mesh: Mesh,
    surface: SurfaceMesh,
    patch_level: PatchLevel,
    retained_modes: np.ndarray,
    previous: OnlineReducedDynamicState,
    residual_blocks: Mapping[int, PatchResidualBlock] | Sequence[PatchResidualBlock],
    load_bases: Sequence[PatchLoadBasis],
    contact_points: np.ndarray,
    penalty: float,
    dt: float,
    sample_areas: np.ndarray | None = None,
    external_force: np.ndarray | None = None,
    contact_offset: float = 0.0,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
    max_iterations: int = 25,
    tolerance: float = 1.0e-8,
    relaxation: float = 1.0,
    include_residual: bool = True,
) -> OnlineReducedDynamicStepResult:
    """Solve one reduced Newmark step with online residual-ILC coupling."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1]")

    Psi = np.asarray(retained_modes, dtype=float)
    if Psi.ndim != 2 or Psi.shape[0] != mesh.n_dofs:
        raise ValueError("retained_modes must have shape (mesh.n_dofs, n_modes)")
    if Psi.shape[1] != previous.state.eta_k.size:
        raise ValueError("previous state eta_k length must match retained_modes columns")

    force_external = np.zeros(mesh.n_dofs, dtype=float) if external_force is None else np.asarray(external_force, dtype=float)
    if force_external.shape != (mesh.n_dofs,):
        raise ValueError("external_force must have shape (mesh.n_dofs,)")

    beta = 0.25
    gamma = 0.5
    c0 = 1.0 / (beta * dt * dt)
    c1 = gamma / (beta * dt)
    q0 = previous.state.eta_k
    qd0 = previous.eta_velocity
    qdd0 = previous.eta_acceleration
    q_predict = q0 + dt * qd0 + dt * dt * (0.5 - beta) * qdd0
    qd_predict = qd0 + dt * (1.0 - gamma) * qdd0

    Kr = np.asarray(Psi.T @ (K @ Psi), dtype=float)
    Mr = np.asarray(Psi.T @ (M @ Psi), dtype=float)
    Cr = rayleigh_alpha * Mr + rayleigh_beta * Kr
    effective = Mr * c0 + Cr * c1 + Kr
    effective = 0.5 * (effective + effective.T)

    q_guess = q_predict.copy()
    active_set = previous.active_set
    iterations: list[OnlineReducedDynamicIteration] = []
    projection: PatchILCProjection | None = None
    samples = ContactSampleSet.empty()
    residual_deformation = np.zeros(mesh.n_dofs, dtype=float)
    converged = False

    for iteration in range(max_iterations):
        residual_deformation = _residual_deformation(residual_blocks, active_set, mesh.n_dofs, include_residual)
        displacement_guess = Psi @ q_guess + residual_deformation
        displaced_surface = surface_with_displacement(surface, displacement_guess)
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
        alpha_update = _active_set_delta_norm(active_set, projection.active_set)
        solve_residual_deformation = _residual_deformation(
            residual_blocks,
            next_active_set,
            mesh.n_dofs,
            include_residual,
        )
        rhs_force = force_external + projection.projected_force_vector - K @ solve_residual_deformation
        reduced_rhs = Psi.T @ rhs_force
        rhs = reduced_rhs - Cr @ qd_predict + (Mr * c0 + Cr * c1) @ q_predict
        q_next = la.solve(effective, rhs, assume_a="sym")
        qdd_next = c0 * (q_next - q_predict)
        qd_next = qd_predict + gamma * dt * qdd_next
        reduced_residual = Mr @ qdd_next + Cr @ qd_next + Kr @ q_next - reduced_rhs
        eta_update = float(np.linalg.norm(q_next - q_guess) / max(1.0, np.linalg.norm(q_next)))
        reduced_residual_norm = float(np.linalg.norm(reduced_residual) / max(1.0, np.linalg.norm(reduced_rhs)))
        iterations.append(
            OnlineReducedDynamicIteration(
                iteration=iteration,
                active_patch_ids=next_active_set.active_patch_ids,
                alpha_update_norm=alpha_update,
                eta_update_norm=eta_update,
                reduced_residual_norm=reduced_residual_norm,
                min_gap=_min_gap(samples),
                normal_force=samples.total_normal_force,
                residual_deformation_norm=float(np.linalg.norm(solve_residual_deformation)),
            )
        )
        q_guess = q_next
        active_set = next_active_set
        if max(alpha_update, eta_update, reduced_residual_norm) <= tolerance:
            converged = True
            break

    residual_deformation = _residual_deformation(residual_blocks, active_set, mesh.n_dofs, include_residual)
    qdd = c0 * (q_guess - q_predict)
    qd = qd_predict + gamma * dt * qdd
    modal_displacement = Psi @ q_guess
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
    next_state = OnlineReducedDynamicState(
        state=ReducedState(previous.state.r, previous.state.theta, q_guess, previous.state.lambda_),
        eta_velocity=qd,
        eta_acceleration=qdd,
        active_set=active_set,
    )
    return OnlineReducedDynamicStepResult(
        previous=previous,
        next=next_state,
        modal_displacement=modal_displacement,
        residual_deformation=residual_deformation,
        displacement=displacement,
        samples=samples,
        projection=projection,
        iterations=tuple(iterations),
        converged=converged,
    )


def baseline_reduced_newmark_step(
    K: sp.spmatrix,
    M: sp.spmatrix,
    retained_modes: np.ndarray,
    previous: OnlineReducedDynamicState,
    dt: float,
    external_force: np.ndarray | None = None,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
) -> OnlineReducedDynamicState:
    """Reduced Newmark step without contact or residual correction."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    Psi = np.asarray(retained_modes, dtype=float)
    force_external = np.zeros(Psi.shape[0], dtype=float) if external_force is None else np.asarray(external_force, dtype=float)
    Kr = np.asarray(Psi.T @ (K @ Psi), dtype=float)
    Mr = np.asarray(Psi.T @ (M @ Psi), dtype=float)
    Cr = rayleigh_alpha * Mr + rayleigh_beta * Kr
    beta = 0.25
    gamma = 0.5
    c0 = 1.0 / (beta * dt * dt)
    c1 = gamma / (beta * dt)
    q_predict = previous.state.eta_k + dt * previous.eta_velocity + dt * dt * (0.5 - beta) * previous.eta_acceleration
    qd_predict = previous.eta_velocity + dt * (1.0 - gamma) * previous.eta_acceleration
    effective = Mr * c0 + Cr * c1 + Kr
    rhs = Psi.T @ force_external - Cr @ qd_predict + (Mr * c0 + Cr * c1) @ q_predict
    q_next = la.solve(0.5 * (effective + effective.T), rhs, assume_a="sym")
    qdd_next = c0 * (q_next - q_predict)
    qd_next = qd_predict + gamma * dt * qdd_next
    return OnlineReducedDynamicState(
        state=ReducedState(previous.state.r, previous.state.theta, q_next, previous.state.lambda_),
        eta_velocity=qd_next,
        eta_acceleration=qdd_next,
        active_set=ActivePatchSet.empty(),
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
