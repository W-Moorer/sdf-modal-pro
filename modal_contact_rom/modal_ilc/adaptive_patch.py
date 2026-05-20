from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.modal_ilc.ilc_projection import ActivePatchSet
from modal_contact_rom.modal_ilc.patch_load_basis import PatchLoadBasis, assemble_load_basis
from modal_contact_rom.modal_ilc.sdf_projection import (
    ContactSampleSet,
    PatchILCProjection,
    assemble_sample_lumped_force_vector,
    detect_sdf_contact_samples,
    total_nodal_force,
)
from modal_contact_rom.surface_patch.extract import SurfaceMesh
from modal_contact_rom.surface_patch.patch_hierarchy import PatchLevel


@dataclass(frozen=True)
class AdaptivePatchActivationConfig:
    neighbor_depth: int = 1
    deactivate_delay: int = 4
    alpha_blend: float = 0.2
    max_active_patches: int | None = None
    projection_error_refine_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.neighbor_depth < 0:
            raise ValueError("neighbor_depth must be non-negative")
        if self.deactivate_delay < 0:
            raise ValueError("deactivate_delay must be non-negative")
        if not (0.0 < self.alpha_blend <= 1.0):
            raise ValueError("alpha_blend must be in (0, 1]")
        if self.max_active_patches is not None and self.max_active_patches < 1:
            raise ValueError("max_active_patches must be positive")


@dataclass(frozen=True)
class AdaptivePatchActivationState:
    active_patch_ids: tuple[int, ...]
    alpha_by_patch: dict[int, float]
    inactive_steps: dict[int, int]

    @classmethod
    def empty(cls) -> AdaptivePatchActivationState:
        return cls(active_patch_ids=(), alpha_by_patch={}, inactive_steps={})


@dataclass(frozen=True)
class AdaptivePatchStepResult:
    step_id: int
    requested_patch_ids: tuple[int, ...]
    active_set: ActivePatchSet
    samples: ContactSampleSet
    projected_force_vector: np.ndarray
    sample_lumped_force_vector: np.ndarray
    projected_normal_force: float
    sample_normal_force: float
    force_jump: float
    alpha_jump: float
    projection_error: float
    estimated_runtime_ratio: float
    state: AdaptivePatchActivationState

    @property
    def active_patch_count(self) -> int:
        return len(self.active_set.active_patch_ids)


@dataclass(frozen=True)
class AdaptivePatchSequenceResult:
    steps: tuple[AdaptivePatchStepResult, ...]
    total_patch_count: int
    full_patch_runtime_ratio: float
    coarse_projection_errors: tuple[float, ...] = ()

    @property
    def max_force_jump(self) -> float:
        if len(self.steps) <= 1:
            return 0.0
        return float(max(step.force_jump for step in self.steps[1:]))

    @property
    def max_alpha_jump(self) -> float:
        if len(self.steps) <= 1:
            return 0.0
        return float(max(step.alpha_jump for step in self.steps[1:]))

    @property
    def max_active_patch_count(self) -> int:
        return max((step.active_patch_count for step in self.steps), default=0)

    @property
    def mean_runtime_ratio(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([step.estimated_runtime_ratio for step in self.steps]))

    @property
    def mean_projection_error(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([step.projection_error for step in self.steps]))

    @property
    def mean_coarse_projection_error(self) -> float:
        if not self.coarse_projection_errors:
            return 0.0
        return float(np.mean(self.coarse_projection_errors))


class AdaptivePatchILCProjector:
    """Stateful Phase-7 active patch smoother for patch ILC projection."""

    def __init__(
        self,
        mesh: Mesh,
        surface: SurfaceMesh,
        patch_level: PatchLevel,
        load_bases: Sequence[PatchLoadBasis],
        config: AdaptivePatchActivationConfig | None = None,
    ) -> None:
        self.mesh = mesh
        self.surface = surface
        self.patch_level = patch_level
        self.load_bases = tuple(load_bases)
        self.config = AdaptivePatchActivationConfig() if config is None else config
        self.state = AdaptivePatchActivationState.empty()
        self._basis_by_patch = {basis.patch_id: basis for basis in self.load_bases}
        self._basis_index_by_patch = {basis.patch_id: index for index, basis in enumerate(self.load_bases)}

    def step(self, samples: ContactSampleSet, step_id: int = 0) -> AdaptivePatchStepResult:
        requested = samples.active_patch_ids
        raw_alpha = _target_alpha_by_patch(samples, self._basis_by_patch)
        raw_error = _projection_error(self.mesh, self.surface, samples, self._basis_by_patch, raw_alpha)
        neighbor_depth = self.config.neighbor_depth
        if raw_error > self.config.projection_error_refine_threshold:
            neighbor_depth = max(neighbor_depth, 1)
        desired = expand_patch_ids_with_neighbors(self.patch_level, requested, depth=neighbor_depth)
        active_patch_ids = self._apply_hysteresis(desired)
        active_patch_ids = self._apply_active_bound(active_patch_ids, raw_alpha, requested)
        alpha_by_patch = self._smooth_alpha(active_patch_ids, raw_alpha, samples.total_normal_force)
        active_bases = tuple(self._basis_by_patch[patch_id] for patch_id in active_patch_ids if patch_id in self._basis_by_patch)
        alpha = np.asarray([alpha_by_patch.get(basis.patch_id, 0.0) for basis in active_bases], dtype=float)
        active_set = ActivePatchSet(
            active_patch_ids=tuple(basis.patch_id for basis in active_bases),
            active_load_basis_indices=tuple(self._basis_index_by_patch[basis.patch_id] for basis in active_bases),
            alpha=alpha,
        )
        projected = assemble_load_basis(active_bases) @ alpha if active_bases else np.zeros(self.mesh.n_dofs, dtype=float)
        sample_lumped = assemble_sample_lumped_force_vector(self.mesh, self.surface, samples)
        previous_force = sum(self.state.alpha_by_patch.values())
        projected_force = float(np.sum(alpha))
        force_jump = _relative_scalar_jump(previous_force, projected_force)
        alpha_jump = _alpha_jump(self.state.alpha_by_patch, alpha_by_patch)
        projection_error = _relative_norm(projected - sample_lumped, sample_lumped)
        next_state = AdaptivePatchActivationState(
            active_patch_ids=active_set.active_patch_ids,
            alpha_by_patch={patch_id: alpha_by_patch.get(patch_id, 0.0) for patch_id in active_set.active_patch_ids},
            inactive_steps=self._next_inactive_steps(desired, active_set.active_patch_ids),
        )
        self.state = next_state
        return AdaptivePatchStepResult(
            step_id=step_id,
            requested_patch_ids=requested,
            active_set=active_set,
            samples=samples,
            projected_force_vector=projected,
            sample_lumped_force_vector=sample_lumped,
            projected_normal_force=projected_force,
            sample_normal_force=samples.total_normal_force,
            force_jump=force_jump,
            alpha_jump=alpha_jump,
            projection_error=projection_error,
            estimated_runtime_ratio=len(active_set.active_patch_ids) / max(1, self.patch_level.patch_count),
            state=next_state,
        )

    def _apply_hysteresis(self, desired: tuple[int, ...]) -> tuple[int, ...]:
        active = list(desired)
        desired_set = set(desired)
        for patch_id in self.state.active_patch_ids:
            if patch_id in desired_set:
                continue
            inactive_count = self.state.inactive_steps.get(patch_id, 0) + 1
            if inactive_count <= self.config.deactivate_delay:
                active.append(patch_id)
        return _unique_tuple(active)

    def _apply_active_bound(
        self,
        active_patch_ids: tuple[int, ...],
        raw_alpha: dict[int, float],
        requested_patch_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        max_active = self.config.max_active_patches
        if max_active is None or len(active_patch_ids) <= max_active:
            return active_patch_ids
        requested = set(requested_patch_ids)
        ranked = sorted(
            active_patch_ids,
            key=lambda patch_id: (
                patch_id not in requested,
                -abs(raw_alpha.get(patch_id, self.state.alpha_by_patch.get(patch_id, 0.0))),
                patch_id,
            ),
        )
        return tuple(sorted(ranked[:max_active]))

    def _smooth_alpha(
        self,
        active_patch_ids: tuple[int, ...],
        raw_alpha: dict[int, float],
        target_total: float,
    ) -> dict[int, float]:
        if not active_patch_ids or target_total <= 0.0:
            return {patch_id: 0.0 for patch_id in active_patch_ids}
        blended: dict[int, float] = {}
        blend = self.config.alpha_blend
        for patch_id in active_patch_ids:
            old_value = self.state.alpha_by_patch.get(patch_id, 0.0)
            target = raw_alpha.get(patch_id, 0.0)
            blended[patch_id] = max((1.0 - blend) * old_value + blend * target, 0.0)
        current_total = sum(blended.values())
        if current_total <= 0.0:
            requested = [patch_id for patch_id in active_patch_ids if raw_alpha.get(patch_id, 0.0) > 0.0]
            if not requested:
                return blended
            share = target_total / float(len(requested))
            for patch_id in requested:
                blended[patch_id] = share
            current_total = target_total
        scale = target_total / max(current_total, 1.0e-30)
        return {patch_id: value * scale for patch_id, value in blended.items()}

    def _next_inactive_steps(
        self,
        desired: tuple[int, ...],
        active_patch_ids: tuple[int, ...],
    ) -> dict[int, int]:
        desired_set = set(desired)
        next_counts: dict[int, int] = {}
        for patch_id in active_patch_ids:
            if patch_id in desired_set:
                next_counts[patch_id] = 0
            else:
                next_counts[patch_id] = self.state.inactive_steps.get(patch_id, 0) + 1
        return next_counts


def expand_patch_ids_with_neighbors(patch_level: PatchLevel, patch_ids: Sequence[int], depth: int = 1) -> tuple[int, ...]:
    """Expand selected patch ids by geodesic one-ring adjacency depth."""

    selected = {int(patch_id) for patch_id in patch_ids}
    frontier = set(selected)
    for _ in range(depth):
        next_frontier: set[int] = set()
        for patch_id in frontier:
            next_frontier.update(patch_level.adjacency.get(int(patch_id), ()))
        next_frontier -= selected
        selected.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(sorted(selected))


def run_adaptive_patch_sequence(
    mesh: Mesh,
    surface: SurfaceMesh,
    patch_level: PatchLevel,
    load_bases: Sequence[PatchLoadBasis],
    frames: Sequence[tuple[np.ndarray, np.ndarray]],
    penalty: float,
    config: AdaptivePatchActivationConfig | None = None,
    coarse_patch_level: PatchLevel | None = None,
    coarse_load_bases: Sequence[PatchLoadBasis] | None = None,
) -> AdaptivePatchSequenceResult:
    projector = AdaptivePatchILCProjector(mesh, surface, patch_level, load_bases, config=config)
    steps: list[AdaptivePatchStepResult] = []
    coarse_errors: list[float] = []
    for step_id, (points, areas) in enumerate(frames):
        samples = detect_sdf_contact_samples(points, surface, patch_level, penalty=penalty, sample_areas=areas)
        steps.append(projector.step(samples, step_id=step_id))
        if coarse_patch_level is not None and coarse_load_bases is not None:
            coarse_samples = detect_sdf_contact_samples(
                points,
                surface,
                coarse_patch_level,
                penalty=penalty,
                sample_areas=areas,
            )
            coarse_alpha = _target_alpha_by_patch(coarse_samples, {basis.patch_id: basis for basis in coarse_load_bases})
            coarse_errors.append(
                _projection_error(
                    mesh,
                    surface,
                    coarse_samples,
                    {basis.patch_id: basis for basis in coarse_load_bases},
                    coarse_alpha,
                )
            )
    return AdaptivePatchSequenceResult(
        steps=tuple(steps),
        total_patch_count=patch_level.patch_count,
        full_patch_runtime_ratio=1.0,
        coarse_projection_errors=tuple(coarse_errors),
    )


def write_adaptive_patch_tables(result: AdaptivePatchSequenceResult, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history_lines = [
        "step,requested_patch_ids,active_patch_ids,active_patch_count,projected_normal_force,"
        "sample_normal_force,projection_error,estimated_runtime_ratio\n"
    ]
    force_lines = ["step,force_jump\n"]
    alpha_lines = ["step,alpha_jump\n"]
    runtime_lines = ["mean_active_runtime_ratio,full_patch_runtime_ratio,max_active_patch_count,total_patch_count\n"]
    for step in result.steps:
        history_lines.append(
            f"{step.step_id},{'|'.join(map(str, step.requested_patch_ids))},"
            f"{'|'.join(map(str, step.active_set.active_patch_ids))},{step.active_patch_count},"
            f"{step.projected_normal_force:.17g},{step.sample_normal_force:.17g},"
            f"{step.projection_error:.17g},{step.estimated_runtime_ratio:.17g}\n"
        )
        force_lines.append(f"{step.step_id},{step.force_jump:.17g}\n")
        alpha_lines.append(f"{step.step_id},{step.alpha_jump:.17g}\n")
    runtime_lines.append(
        f"{result.mean_runtime_ratio:.17g},{result.full_patch_runtime_ratio:.17g},"
        f"{result.max_active_patch_count},{result.total_patch_count}\n"
    )
    summary = (
        "# Adaptive Patch Summary\n\n"
        f"Steps: {len(result.steps)}\n"
        f"Max force jump: {result.max_force_jump:.6g}\n"
        f"Max alpha jump: {result.max_alpha_jump:.6g}\n"
        f"Max active patch count: {result.max_active_patch_count}\n"
        f"Total patch count: {result.total_patch_count}\n"
        f"Mean active runtime ratio: {result.mean_runtime_ratio:.6g}\n"
        f"Mean projection error: {result.mean_projection_error:.6g}\n"
        f"Mean coarse-only projection error: {result.mean_coarse_projection_error:.6g}\n"
    )
    (output_path / "active_patch_history.csv").write_text("".join(history_lines), encoding="utf-8")
    (output_path / "force_jump.csv").write_text("".join(force_lines), encoding="utf-8")
    (output_path / "alpha_jump.csv").write_text("".join(alpha_lines), encoding="utf-8")
    (output_path / "runtime_vs_active_patch.csv").write_text("".join(runtime_lines), encoding="utf-8")
    (output_path / "adaptive_patch_summary.md").write_text(summary, encoding="utf-8")


def _target_alpha_by_patch(samples: ContactSampleSet, basis_by_patch: dict[int, PatchLoadBasis]) -> dict[int, float]:
    target: dict[int, float] = {}
    for patch_id in samples.active_patch_ids:
        basis = basis_by_patch.get(int(patch_id))
        if basis is None:
            continue
        direction = basis.resultant if basis.resultant is not None else total_nodal_force(basis.B_j[:, 0])
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            continue
        direction = direction / norm
        mask = samples.patch_ids == int(patch_id)
        target[int(patch_id)] = max(float(np.sum(samples.forces[mask] @ direction)), 0.0)
    return target


def _projection_error(
    mesh: Mesh,
    surface: SurfaceMesh,
    samples: ContactSampleSet,
    basis_by_patch: dict[int, PatchLoadBasis],
    alpha_by_patch: dict[int, float],
) -> float:
    if samples.sample_count == 0:
        return 0.0
    active_bases = tuple(basis_by_patch[patch_id] for patch_id in sorted(alpha_by_patch) if patch_id in basis_by_patch)
    alpha = np.asarray([alpha_by_patch[basis.patch_id] for basis in active_bases], dtype=float)
    projected = assemble_load_basis(active_bases) @ alpha if active_bases else np.zeros(mesh.n_dofs, dtype=float)
    sample_lumped = assemble_sample_lumped_force_vector(mesh, surface, samples)
    return _relative_norm(projected - sample_lumped, sample_lumped)


def _relative_norm(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value) / max(float(np.linalg.norm(reference)), 1.0e-30))


def _relative_scalar_jump(previous: float, current: float) -> float:
    if abs(previous) <= 1.0e-30:
        return 0.0
    return abs(current - previous) / max(abs(previous), abs(current), 1.0e-30)


def _alpha_jump(previous: dict[int, float], current: dict[int, float]) -> float:
    if not previous:
        return 0.0
    patch_ids = sorted(set(previous) | set(current))
    old = np.asarray([previous.get(patch_id, 0.0) for patch_id in patch_ids], dtype=float)
    new = np.asarray([current.get(patch_id, 0.0) for patch_id in patch_ids], dtype=float)
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(old), np.linalg.norm(new), 1.0e-30))


def _unique_tuple(values: Sequence[int]) -> tuple[int, ...]:
    unique: list[int] = []
    for value in values:
        item = int(value)
        if item not in unique:
            unique.append(item)
    return tuple(sorted(unique))
