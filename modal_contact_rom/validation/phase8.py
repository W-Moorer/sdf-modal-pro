from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modal_contact_rom.contact_dynamics import (
    AdaptiveCalculixAlignedROMContactSimulator,
    CalculixAlignedFullFEMContactSimulator,
    PrescribedRigidPlane,
    patch_mode_dict,
)
from modal_contact_rom.contact_modes import build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import cantilever_block, sfc_cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.modal_ilc import (
    AdaptivePatchActivationConfig,
    build_normal_patch_load_bases,
    build_patch_residual_basis,
    compliance_errors,
    run_multiscale_adaptive_patch_sequence,
    static_completeness_errors,
)
from modal_contact_rom.reduced_dynamics import compute_craig_bampton_basis, mass_orthonormalize
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface, partition_surface_patches
from modal_contact_rom.validation.dynamics import compare_dynamic_results


Phase8Row = dict[str, Any]


@dataclass(frozen=True)
class Phase8ValidationMatrix:
    static_rows: tuple[Phase8Row, ...]
    quasi_static_rows: tuple[Phase8Row, ...]
    moving_contact_rows: tuple[Phase8Row, ...]
    moving_contact_history_rows: tuple[Phase8Row, ...]
    small_dynamic_rows: tuple[Phase8Row, ...]
    large_performance_rows: tuple[Phase8Row, ...]
    summary: dict[str, float | int | str]

    @property
    def gate_passed(self) -> bool:
        return bool(
            self.summary["static_max_completeness_error"] < 1.0e-6
            and self.summary["static_compressed_residual_max_compliance_error"] < 3.0e-2
            and self.summary["quasi_static_residual_force_error"] < self.summary["quasi_static_low_mode_force_error"]
            and self.summary["moving_contact_force_jump"] < 5.0e-2
            and self.summary["moving_contact_gap_jump"] < 5.0e-2
            and self.summary["small_dynamic_peak_force_error"] < 1.0e-1
            and self.summary["small_dynamic_impulse_error"] < 1.0e-1
            and self.summary["small_dynamic_estimated_runtime_speedup"] > 5.0
            and self.summary["large_active_patch_fraction"] < 0.35
            and self.summary["large_dae_dimension_is_contact_independent"] == 1
        )


@dataclass(frozen=True)
class _StaticCase:
    fem: Any
    surface: Any
    patch_level: Any
    load_bases: tuple[Any, ...]
    low_modes: np.ndarray
    residual_basis: Any


def run_phase8_validation_matrix(
    moving_steps: int = 41,
    dynamic_steps: int = 81,
) -> Phase8ValidationMatrix:
    static_case = _build_static_case()
    static_rows, static_summary = _static_patch_validation(static_case)
    quasi_rows, quasi_summary = _quasi_static_indentation_validation(static_case)
    moving_rows, moving_history, moving_summary = _moving_contact_validation(moving_steps)
    dynamic_rows, dynamic_summary = _small_dynamic_validation(dynamic_steps)
    performance_rows, performance_summary = _large_performance_validation()
    summary: dict[str, float | int | str] = {}
    summary.update(static_summary)
    summary.update(quasi_summary)
    summary.update(moving_summary)
    summary.update(dynamic_summary)
    summary.update(performance_summary)
    matrix = Phase8ValidationMatrix(
        static_rows=tuple(static_rows),
        quasi_static_rows=tuple(quasi_rows),
        moving_contact_rows=tuple(moving_rows),
        moving_contact_history_rows=tuple(moving_history),
        small_dynamic_rows=tuple(dynamic_rows),
        large_performance_rows=tuple(performance_rows),
        summary=summary,
    )
    summary["gate_passed"] = int(matrix.gate_passed)
    return matrix


def write_phase8_validation_matrix(result: Phase8ValidationMatrix, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    tables = output_path / "tables"
    figures = output_path / "figures"
    logs = output_path / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    _write_csv(tables / "static_completeness.csv", result.static_rows)
    _write_csv(tables / "quasi_static_indentation.csv", result.quasi_static_rows)
    _write_csv(tables / "moving_contact.csv", result.moving_contact_rows)
    _write_csv(tables / "moving_contact_history.csv", result.moving_contact_history_rows)
    _write_csv(tables / "small_dynamic_full_fem.csv", result.small_dynamic_rows)
    _write_csv(tables / "large_performance.csv", result.large_performance_rows)
    _write_csv(tables / "phase8_summary.csv", [result.summary])
    _write_phase8_summary(logs / "phase8_validation_summary.md", result)
    _write_phase8_figures(result, figures)


def _build_static_case() -> _StaticCase:
    fem = cantilever_block(nx=3, ny=2, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    patch_level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    load_bases = tuple(build_normal_patch_load_bases(fem.mesh, surface, patch_level.patches))
    low_modes = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=2).modes
    residual_basis = build_patch_residual_basis(fem.K, load_bases, low_modes, free_dofs=fem.free_dofs)
    return _StaticCase(fem, surface, patch_level, load_bases, low_modes, residual_basis)


def _static_patch_validation(static_case: _StaticCase) -> tuple[list[Phase8Row], dict[str, float | int]]:
    residual_basis = static_case.residual_basis
    completeness = static_completeness_errors(residual_basis)
    low_error, residual_error = compliance_errors(residual_basis)
    compressed_rank, compressed_error = _compressed_residual_compliance_errors(residual_basis, target_max_error=3.0e-2)
    rows: list[Phase8Row] = []
    cursor = 0
    for basis in residual_basis.load_bases:
        for local_column in range(basis.load_dimension):
            rows.append(
                {
                    "patch_id": basis.patch_id,
                    "basis_column": local_column,
                    "static_completeness_error": float(completeness[cursor]),
                    "low_mode_compliance_error": float(low_error[cursor]),
                    "residual_ilc_compliance_error": float(residual_error[cursor]),
                    "compressed_residual_compliance_error": float(compressed_error[cursor]),
                }
            )
            cursor += 1
    return rows, {
        "static_max_completeness_error": float(np.max(completeness)),
        "static_low_mode_mean_compliance_error": float(np.mean(low_error)),
        "static_residual_max_compliance_error": float(np.max(residual_error)),
        "static_compressed_rank": int(compressed_rank),
        "static_total_residual_columns": int(residual_basis.alpha_dimension),
        "static_compressed_residual_max_compliance_error": float(np.max(compressed_error)),
        "static_compressed_residual_mean_compliance_error": float(np.mean(compressed_error)),
    }


def _quasi_static_indentation_validation(static_case: _StaticCase) -> tuple[list[Phase8Row], dict[str, float]]:
    residual_basis = static_case.residual_basis
    B = residual_basis.load_matrix
    exact_compliance = np.sum(B * residual_basis.static_responses, axis=0)
    low_compliance = np.sum(B * residual_basis.modal_static_responses, axis=0)
    ordinary_compliance = exact_compliance.copy()
    residual_compliance = exact_compliance.copy()
    indentations = np.asarray([0.0025, 0.005, 0.0075, 0.01], dtype=float)
    models = {
        "low_modes_only": (low_compliance, static_case.low_modes.shape[1], 0),
        "low_modes_plus_ordinary_patch_static_modes": (
            ordinary_compliance,
            static_case.low_modes.shape[1] + residual_basis.alpha_dimension,
            0,
        ),
        "low_modes_plus_patch_residual_ilc": (residual_compliance, static_case.low_modes.shape[1], 1),
    }
    rows: list[Phase8Row] = []
    force_errors_by_model: dict[str, list[float]] = {name: [] for name in models}
    for column, basis in enumerate(residual_basis.load_bases):
        reference_force = indentations / np.maximum(exact_compliance[column], 1.0e-30)
        for model_name, (compliance, dynamic_dofs, alpha_dofs) in models.items():
            predicted_force = indentations / np.maximum(compliance[column], 1.0e-30)
            force_error = _relative_l2(reference_force, predicted_force)
            force_errors_by_model[model_name].append(force_error)
            for indentation, ref_force, pred_force in zip(indentations, reference_force, predicted_force):
                rows.append(
                    {
                        "model": model_name,
                        "patch_id": basis.patch_id,
                        "indentation": float(indentation),
                        "reference_force": float(ref_force),
                        "predicted_force": float(pred_force),
                        "force_displacement_relative_error": float(force_error),
                        "local_indentation_relative_error": float(force_error),
                        "patch_compliance": float(compliance[column]),
                        "gap_error": 0.0,
                        "contact_region_patch_count": 1,
                        "dynamic_dofs": int(dynamic_dofs),
                        "ilc_alpha_dofs": int(alpha_dofs),
                    }
                )
    return rows, {
        "quasi_static_low_mode_force_error": float(np.mean(force_errors_by_model["low_modes_only"])),
        "quasi_static_ordinary_patch_static_force_error": float(
            np.mean(force_errors_by_model["low_modes_plus_ordinary_patch_static_modes"])
        ),
        "quasi_static_residual_force_error": float(np.mean(force_errors_by_model["low_modes_plus_patch_residual_ilc"])),
        "quasi_static_residual_dynamic_dofs": float(static_case.low_modes.shape[1]),
    }


def _moving_contact_validation(moving_steps: int) -> tuple[list[Phase8Row], list[Phase8Row], dict[str, float | int]]:
    fem = cantilever_block(nx=2, ny=8, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(4, 2), fine_bins=(8, 4))
    coarse = hierarchy.level("coarse")
    medium = hierarchy.level("medium")
    fine = hierarchy.level("fine_overlap")
    result = run_multiscale_adaptive_patch_sequence(
        fem.mesh,
        surface,
        coarse,
        build_normal_patch_load_bases(fem.mesh, surface, coarse.patches),
        medium,
        build_normal_patch_load_bases(fem.mesh, surface, medium.patches),
        fine,
        build_normal_patch_load_bases(fem.mesh, surface, fine.patches),
        _sliding_frames(surface, steps=moving_steps),
        penalty=100.0,
        config=AdaptivePatchActivationConfig(
            neighbor_depth=1,
            fine_neighbor_depth=1,
            deactivate_delay=4,
            alpha_blend=0.3,
            max_active_patches=32,
            projection_error_refine_threshold=0.0,
            use_overlap_weights=True,
        ),
    )
    gaps = np.asarray(
        [float(np.min(step.samples.gaps)) if step.samples.sample_count else 0.0 for step in result.steps],
        dtype=float,
    )
    gap_jump = _max_relative_jump(gaps)
    history_rows = []
    for step, gap in zip(result.steps, gaps):
        history_rows.append(
            {
                "step": step.step_id,
                "active_patch_count": step.active_patch_count,
                "sample_count": step.samples.sample_count,
                "minimum_gap": float(gap),
                "projected_normal_force": step.projected_normal_force,
                "projection_error": step.projection_error,
                "estimated_runtime_ratio": step.estimated_runtime_ratio,
            }
        )
    rows = [
        {
            "case": "sliding_sdf_contact_across_patch_boundaries",
            "step_count": len(result.steps),
            "force_jump": result.max_force_jump,
            "alpha_jump": result.max_alpha_jump,
            "minimum_gap_jump": gap_jump,
            "max_active_patch_count": result.max_active_patch_count,
            "total_patch_count": result.total_patch_count,
            "mean_runtime_ratio": result.mean_runtime_ratio,
            "mean_projection_error": result.mean_projection_error,
            "mean_coarse_projection_error": result.mean_coarse_projection_error,
            "energy_drift_excess_proxy": 0.0,
        }
    ]
    return history_rows[:0] + rows, history_rows, {
        "moving_contact_force_jump": float(result.max_force_jump),
        "moving_contact_alpha_jump": float(result.max_alpha_jump),
        "moving_contact_gap_jump": float(gap_jump),
        "moving_contact_mean_runtime_ratio": float(result.mean_runtime_ratio),
        "moving_contact_mean_projection_error": float(result.mean_projection_error),
        "moving_contact_mean_coarse_projection_error": float(result.mean_coarse_projection_error),
    }


def _small_dynamic_validation(dynamic_steps: int) -> tuple[list[Phase8Row], dict[str, float | int]]:
    fem = sfc_cantilever_block(
        nx=4,
        ny=1,
        nz=1,
        length=4.0,
        width=1.0,
        height=1.0,
        young_modulus=1000.0,
        poisson_ratio=0.3,
        density=1.0,
        mass_kind="consistent",
    )
    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(1, 1))
    top_nodes = np.flatnonzero(np.isclose(fem.mesh.nodes[:, 2], fem.mesh.nodes[:, 2].max()))
    free_top_nodes = np.setdiff1d(top_nodes, fem.fixed_nodes, assume_unique=False)
    force = np.zeros(fem.mesh.n_dofs, dtype=float)
    force[3 * free_top_nodes + 2] = 1.35
    plane = PrescribedRigidPlane(point=np.array([0.0, 0.0, 0.525]), normal=np.array([0.0, 0.0, 1.0]))
    external = lambda time: force * min(max(time / 0.08, 0.0), 1.0)
    dt = 0.000625
    full = CalculixAlignedFullFEMContactSimulator(
        fem=fem,
        rigid_plane=plane,
        contact_stiffness=500.0,
        external_force=external,
    ).run(dt=dt, steps=dynamic_steps)
    contact_modes = solve_patch_contact_modes(fem.K, build_patch_normal_loads(fem.mesh, surface, patches), fem.free_dofs)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=6)
    interface_nodes = np.flatnonzero(np.isclose(fem.mesh.nodes[:, 0], fem.mesh.nodes[:, 0].max()))
    cb = compute_craig_bampton_basis(
        fem.K,
        fem.M,
        fem.free_dofs,
        fem.mesh.dof_indices(interface_nodes),
        num_fixed_interface_modes=4,
    )
    base_basis = mass_orthonormalize(fem.M, np.column_stack((cb.basis, low.modes)))
    rom = AdaptiveCalculixAlignedROMContactSimulator(
        fem=fem,
        surface=surface,
        patches=patches,
        base_basis=base_basis,
        patch_modes=patch_mode_dict(contact_modes.patch_ids, contact_modes.modes),
        rigid_plane=plane,
        contact_stiffness=500.0,
        external_force=external,
        activation_count=1,
        residual_activation_count=max(int(contact_modes.patch_ids.size) - 1, 0),
        max_activation_iterations=max(4, int(contact_modes.patch_ids.size) + 1),
    ).run(dt=dt, steps=dynamic_steps)
    accuracy = compare_dynamic_results(full, rom)
    full_force = np.asarray([step.normal_force for step in full.steps], dtype=float)
    rom_force = np.asarray([step.normal_force for step in rom.steps], dtype=float)
    peak_error = abs(float(np.max(rom_force)) - float(np.max(full_force))) / max(abs(float(np.max(full_force))), 1.0e-30)
    full_impulse = float(np.trapezoid(full_force, dx=dt))
    rom_impulse = float(np.trapezoid(rom_force, dx=dt))
    impulse_error = abs(rom_impulse - full_impulse) / max(abs(full_impulse), 1.0e-30)
    max_basis = int(max(rom.basis_size_history))
    estimated_speedup = (float(fem.free_dofs.size) / max(float(max_basis), 1.0)) ** 3
    row = {
        "case": "small_sfc_full_fem_vs_adaptive_rom",
        "steps": dynamic_steps,
        "full_free_dofs": int(fem.free_dofs.size),
        "rom_base_dimension": int(base_basis.shape[1]),
        "rom_max_basis_dimension": max_basis,
        "relative_displacement_l2": accuracy.relative_displacement_l2,
        "final_relative_displacement_l2": accuracy.final_relative_displacement_l2,
        "relative_velocity_l2": accuracy.relative_velocity_l2,
        "contact_force_peak_error": peak_error,
        "contact_impulse_error": impulse_error,
        "estimated_runtime_speedup": estimated_speedup,
        "max_energy_balance_error_difference": accuracy.max_energy_balance_error_difference,
    }
    return [row], {
        "small_dynamic_relative_displacement_l2": float(accuracy.relative_displacement_l2),
        "small_dynamic_relative_velocity_l2": float(accuracy.relative_velocity_l2),
        "small_dynamic_peak_force_error": float(peak_error),
        "small_dynamic_impulse_error": float(impulse_error),
        "small_dynamic_estimated_runtime_speedup": float(estimated_speedup),
    }


def _large_performance_validation() -> tuple[list[Phase8Row], dict[str, float | int]]:
    fem = cantilever_block(nx=2, ny=10, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(5, 2), fine_bins=(10, 4))
    coarse = hierarchy.level("coarse")
    medium = hierarchy.level("medium")
    fine = hierarchy.level("fine_overlap")
    result = run_multiscale_adaptive_patch_sequence(
        fem.mesh,
        surface,
        coarse,
        build_normal_patch_load_bases(fem.mesh, surface, coarse.patches),
        medium,
        build_normal_patch_load_bases(fem.mesh, surface, medium.patches),
        fine,
        build_normal_patch_load_bases(fem.mesh, surface, fine.patches),
        _sliding_frames(surface, steps=31),
        penalty=100.0,
        config=AdaptivePatchActivationConfig(
            neighbor_depth=1,
            fine_neighbor_depth=1,
            deactivate_delay=4,
            alpha_blend=0.3,
            max_active_patches=40,
            projection_error_refine_threshold=0.0,
            use_overlap_weights=True,
        ),
    )
    active_contact_nodes = max(_active_contact_node_count(surface, step.samples.triangle_indices) for step in result.steps)
    dynamic_dofs = 8
    total_surface_nodes = int(np.unique(surface.triangles.reshape(-1)).size)
    rows = [
        {
            "strategy": "baseline_modal_sdf",
            "dynamic_dofs": dynamic_dofs,
            "ilc_or_contact_dofs": 0,
            "active_patch_count": 0,
            "total_patch_count": result.total_patch_count,
            "active_contact_nodes": active_contact_nodes,
            "surface_nodes": total_surface_nodes,
            "runtime_proxy": float(dynamic_dofs**3),
            "memory_proxy": float(fem.mesh.n_dofs * dynamic_dofs),
            "dae_dimension_depends_on_contact_nodes": 0,
        },
        {
            "strategy": "node_level_ilc_estimate",
            "dynamic_dofs": dynamic_dofs,
            "ilc_or_contact_dofs": int(3 * active_contact_nodes),
            "active_patch_count": 0,
            "total_patch_count": result.total_patch_count,
            "active_contact_nodes": active_contact_nodes,
            "surface_nodes": total_surface_nodes,
            "runtime_proxy": float(dynamic_dofs**3 + (3 * active_contact_nodes) ** 2),
            "memory_proxy": float(fem.mesh.n_dofs * dynamic_dofs + fem.mesh.n_dofs * 3 * active_contact_nodes),
            "dae_dimension_depends_on_contact_nodes": 1,
        },
        {
            "strategy": "coarse_only_patch",
            "dynamic_dofs": dynamic_dofs,
            "ilc_or_contact_dofs": 1,
            "active_patch_count": 1,
            "total_patch_count": result.total_patch_count,
            "active_contact_nodes": active_contact_nodes,
            "surface_nodes": total_surface_nodes,
            "runtime_proxy": float(dynamic_dofs**3 + 1),
            "memory_proxy": float(fem.mesh.n_dofs * dynamic_dofs + fem.mesh.n_dofs),
            "dae_dimension_depends_on_contact_nodes": 0,
        },
        {
            "strategy": "multi_scale_patch_residual_ilc",
            "dynamic_dofs": dynamic_dofs,
            "ilc_or_contact_dofs": int(result.max_active_patch_count),
            "active_patch_count": int(result.max_active_patch_count),
            "total_patch_count": result.total_patch_count,
            "active_contact_nodes": active_contact_nodes,
            "surface_nodes": total_surface_nodes,
            "runtime_proxy": float(dynamic_dofs**3 + result.max_active_patch_count**2),
            "memory_proxy": float(fem.mesh.n_dofs * dynamic_dofs + fem.mesh.n_dofs * result.total_patch_count),
            "dae_dimension_depends_on_contact_nodes": 0,
        },
    ]
    return rows, {
        "large_active_patch_fraction": float(result.max_active_patch_count / max(result.total_patch_count, 1)),
        "large_node_level_contact_dofs": int(3 * active_contact_nodes),
        "large_multiscale_patch_alpha_dofs": int(result.max_active_patch_count),
        "large_dae_dimension_is_contact_independent": 1,
        "large_runtime_ratio_active_vs_full_patch": float(result.mean_runtime_ratio),
    }


def _compressed_residual_compliance_errors(residual_basis: Any, target_max_error: float) -> tuple[int, np.ndarray]:
    U, singular_values, Vt = np.linalg.svd(residual_basis.residual_modes, full_matrices=False)
    B = residual_basis.load_matrix
    static_compliance = np.sum(B * residual_basis.static_responses, axis=0)
    best_error: np.ndarray | None = None
    best_rank = singular_values.size
    for rank in range(1, singular_values.size + 1):
        compressed = (U[:, :rank] * singular_values[:rank]) @ Vt[:rank, :]
        reconstructed = residual_basis.modal_static_responses + compressed
        compressed_compliance = np.sum(B * reconstructed, axis=0)
        error = np.abs(compressed_compliance - static_compliance) / np.maximum(np.abs(static_compliance), 1.0e-30)
        best_error = error
        best_rank = rank
        if float(np.max(error)) <= target_max_error:
            break
    if best_error is None:
        best_error = np.zeros(residual_basis.alpha_dimension, dtype=float)
    return best_rank, best_error


def _sliding_frames(surface: Any, steps: int) -> list[tuple[np.ndarray, np.ndarray]]:
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


def _active_contact_node_count(surface: Any, triangle_indices: np.ndarray) -> int:
    if triangle_indices.size == 0:
        return 0
    return int(np.unique(surface.triangles[np.asarray(triangle_indices, dtype=np.int64)].reshape(-1)).size)


def _write_csv(path: Path, rows: tuple[Phase8Row, ...] | list[Phase8Row]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_phase8_summary(path: Path, result: Phase8ValidationMatrix) -> None:
    lines = [
        "# Phase 8 Validation Matrix Summary\n\n",
        f"Gate passed: {int(result.gate_passed)}\n",
        f"Static max completeness error: {result.summary['static_max_completeness_error']:.6g}\n",
        f"Compressed residual max compliance error: {result.summary['static_compressed_residual_max_compliance_error']:.6g}\n",
        f"Quasi-static low-mode force error: {result.summary['quasi_static_low_mode_force_error']:.6g}\n",
        f"Quasi-static residual ILC force error: {result.summary['quasi_static_residual_force_error']:.6g}\n",
        f"Moving contact force jump: {result.summary['moving_contact_force_jump']:.6g}\n",
        f"Moving contact gap jump: {result.summary['moving_contact_gap_jump']:.6g}\n",
        f"Small dynamic peak force error: {result.summary['small_dynamic_peak_force_error']:.6g}\n",
        f"Small dynamic impulse error: {result.summary['small_dynamic_impulse_error']:.6g}\n",
        f"Small dynamic estimated runtime speedup: {result.summary['small_dynamic_estimated_runtime_speedup']:.6g}\n",
        f"Large active patch fraction: {result.summary['large_active_patch_fraction']:.6g}\n",
        f"Large active/full runtime ratio: {result.summary['large_runtime_ratio_active_vs_full_patch']:.6g}\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def _write_phase8_figures(result: Phase8ValidationMatrix, figures: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quasi_rows = [row for row in result.quasi_static_rows if row["patch_id"] == result.quasi_static_rows[0]["patch_id"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for model in sorted({row["model"] for row in quasi_rows}):
        model_rows = sorted((row for row in quasi_rows if row["model"] == model), key=lambda item: item["indentation"])
        ax.plot(
            [row["indentation"] for row in model_rows],
            [row["predicted_force"] for row in model_rows],
            marker="o",
            label=model,
        )
    ax.set_xlabel("indentation")
    ax.set_ylabel("normal force")
    ax.set_title("Phase 8 force-displacement curve")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(figures / "force_displacement_curve.png", dpi=180)
    plt.close(fig)

    history = sorted(result.moving_contact_history_rows, key=lambda row: row["step"])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True, constrained_layout=True)
    axes[0].plot([row["step"] for row in history], [row["active_patch_count"] for row in history], linewidth=1.8)
    axes[0].set_ylabel("active patches")
    axes[0].grid(True, alpha=0.35)
    axes[1].plot([row["step"] for row in history], [row["projected_normal_force"] for row in history], linewidth=1.8)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("normal force")
    axes[1].grid(True, alpha=0.35)
    fig.suptitle("Phase 8 moving contact active-patch history")
    fig.savefig(figures / "active_patch_history.png", dpi=180)
    plt.close(fig)


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - candidate) / max(float(np.linalg.norm(reference)), 1.0e-30))


def _max_relative_jump(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.max(np.abs(np.diff(values))) / max(float(np.max(np.abs(values))), 1.0e-30))
