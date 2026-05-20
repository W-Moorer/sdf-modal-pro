from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.modal_ilc import (
    AdaptivePatchActivationConfig,
    build_normal_patch_load_bases,
    contact_samples_from_sdf_query,
    detect_sdf_contact_samples,
    project_contact_samples_to_patch_ilc,
    run_adaptive_patch_sequence,
    run_multiscale_adaptive_patch_sequence,
)
from modal_contact_rom.sdf_query.mesh_distance import _closest_point_on_triangle, query_signed_distances
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface
from modal_contact_rom.validation.phase8 import Phase8ValidationMatrix, run_phase8_validation_matrix


Phase9Row = dict[str, Any]


@dataclass(frozen=True)
class Phase9ClaimGate:
    phase8: Phase8ValidationMatrix
    static_completeness_rows: tuple[Phase9Row, ...]
    compliance_rows: tuple[Phase9Row, ...]
    contact_force_rows: tuple[Phase9Row, ...]
    runtime_rows: tuple[Phase9Row, ...]
    ablation_rows: tuple[Phase9Row, ...]
    sdf_projection_rows: tuple[Phase9Row, ...]
    complex_surface_rows: tuple[Phase9Row, ...]
    performance_optimization_rows: tuple[Phase9Row, ...]
    three_way_external_rows: tuple[Phase9Row, ...]
    claim_rows: tuple[Phase9Row, ...]
    active_patch_history_rows: tuple[Phase9Row, ...]
    summary: dict[str, float | int | str]

    @property
    def gate_passed(self) -> bool:
        return bool(self.summary["phase9_gate_passed"])


def run_phase9_claim_gate() -> Phase9ClaimGate:
    phase8 = run_phase8_validation_matrix()
    single_scale = _single_scale_sdf_active_ablation()
    static_rows = _static_completeness_rows(phase8)
    compliance_rows = _compliance_rows(phase8)
    contact_rows = _contact_force_rows(phase8, single_scale)
    runtime_rows = _runtime_rows(phase8)
    ablation_rows = _ablation_rows(phase8, single_scale)
    sdf_projection_rows = _sdf_projection_evidence_rows()
    complex_surface_rows, performance_rows = _complex_surface_validation_and_performance_rows()
    three_way_external_rows = _three_way_external_evidence_rows()
    claim_rows = _claim_rows(
        phase8,
        single_scale,
        sdf_projection_rows,
        complex_surface_rows,
        performance_rows,
        three_way_external_rows,
    )
    summary = _summary(
        phase8,
        single_scale,
        sdf_projection_rows,
        complex_surface_rows,
        performance_rows,
        three_way_external_rows,
        claim_rows,
    )
    return Phase9ClaimGate(
        phase8=phase8,
        static_completeness_rows=tuple(static_rows),
        compliance_rows=tuple(compliance_rows),
        contact_force_rows=tuple(contact_rows),
        runtime_rows=tuple(runtime_rows),
        ablation_rows=tuple(ablation_rows),
        sdf_projection_rows=tuple(sdf_projection_rows),
        complex_surface_rows=tuple(complex_surface_rows),
        performance_optimization_rows=tuple(performance_rows),
        three_way_external_rows=tuple(three_way_external_rows),
        claim_rows=tuple(claim_rows),
        active_patch_history_rows=tuple(phase8.moving_contact_history_rows),
        summary=summary,
    )


def write_phase9_claim_gate(result: Phase9ClaimGate, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    tables = output_path / "tables"
    figures = output_path / "figures"
    logs = output_path / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    _write_csv(tables / "static_completeness.csv", result.static_completeness_rows)
    _write_csv(tables / "compliance_error.csv", result.compliance_rows)
    _write_csv(tables / "contact_force_error.csv", result.contact_force_rows)
    _write_csv(tables / "runtime_scaling.csv", result.runtime_rows)
    _write_csv(tables / "ablation_summary.csv", result.ablation_rows)
    _write_csv(tables / "sdf_patch_projection.csv", result.sdf_projection_rows)
    _write_csv(tables / "complex_surface_moving_contact.csv", result.complex_surface_rows)
    _write_csv(tables / "performance_optimization.csv", result.performance_optimization_rows)
    _write_csv(tables / "three_way_external_fem.csv", result.three_way_external_rows)
    _write_csv(tables / "claim_gate.csv", result.claim_rows)
    _write_csv(tables / "active_patch_history.csv", result.active_patch_history_rows)
    _write_claims_markdown(logs / "patch_ilc_claims.md", result)
    _write_phase9_figures(result, figures)
    _copy_three_way_figures(result.three_way_external_rows, figures)


def _single_scale_sdf_active_ablation() -> dict[str, float | int]:
    fem = cantilever_block(nx=2, ny=8, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(4, 2))
    coarse = hierarchy.level("coarse")
    medium = hierarchy.level("medium")
    result = run_adaptive_patch_sequence(
        fem.mesh,
        surface,
        medium,
        build_normal_patch_load_bases(fem.mesh, surface, medium.patches),
        _sliding_frames(surface, steps=41),
        penalty=100.0,
        config=AdaptivePatchActivationConfig(
            neighbor_depth=1,
            deactivate_delay=4,
            alpha_blend=0.3,
            max_active_patches=8,
        ),
        coarse_patch_level=coarse,
        coarse_load_bases=build_normal_patch_load_bases(fem.mesh, surface, coarse.patches),
    )
    unique_requested = len({patch_id for step in result.steps for patch_id in step.requested_patch_ids})
    return {
        "mean_projection_error": float(result.mean_projection_error),
        "mean_coarse_projection_error": float(result.mean_coarse_projection_error),
        "max_force_jump": float(result.max_force_jump),
        "max_alpha_jump": float(result.max_alpha_jump),
        "max_active_patch_count": int(result.max_active_patch_count),
        "total_patch_count": int(result.total_patch_count),
        "active_patch_fraction": float(result.max_active_patch_count / max(result.total_patch_count, 1)),
        "mean_runtime_ratio": float(result.mean_runtime_ratio),
        "unique_requested_patch_count": int(unique_requested),
    }


def _static_completeness_rows(phase8: Phase8ValidationMatrix) -> list[Phase9Row]:
    return [
        {
            "patch_id": row["patch_id"],
            "basis_column": row["basis_column"],
            "static_completeness_error": row["static_completeness_error"],
            "gate_threshold": 1.0e-6,
            "passed": int(float(row["static_completeness_error"]) < 1.0e-6),
        }
        for row in phase8.static_rows
    ]


def _compliance_rows(phase8: Phase8ValidationMatrix) -> list[Phase9Row]:
    rows: list[Phase9Row] = []
    for row in phase8.static_rows:
        rows.append(
            {
                "patch_id": row["patch_id"],
                "basis_column": row["basis_column"],
                "low_modes_only_error": row["low_mode_compliance_error"],
                "patch_residual_ilc_error": row["residual_ilc_compliance_error"],
                "compressed_patch_residual_error": row["compressed_residual_compliance_error"],
            }
        )
    return rows


def _contact_force_rows(phase8: Phase8ValidationMatrix, single_scale: dict[str, float | int]) -> list[Phase9Row]:
    return [
        {
            "case": "quasi_static_indentation",
            "low_modes_only_error": phase8.summary["quasi_static_low_mode_force_error"],
            "ordinary_patch_static_modes_error": phase8.summary["quasi_static_ordinary_patch_static_force_error"],
            "patch_residual_ilc_error": phase8.summary["quasi_static_residual_force_error"],
        },
        {
            "case": "sdf_moving_contact_projection",
            "coarse_only_error": single_scale["mean_coarse_projection_error"],
            "single_scale_sdf_active_error": single_scale["mean_projection_error"],
            "multi_scale_sdf_active_error": phase8.summary["moving_contact_mean_projection_error"],
        },
        {
            "case": "small_full_fem_dynamic",
            "displacement_rmse": phase8.summary["small_dynamic_displacement_rmse"],
            "velocity_rmse": phase8.summary["small_dynamic_velocity_rmse"],
            "contact_force_peak_error": phase8.summary["small_dynamic_peak_force_error"],
            "contact_impulse_error": phase8.summary["small_dynamic_impulse_error"],
        },
    ]


def _runtime_rows(phase8: Phase8ValidationMatrix) -> list[Phase9Row]:
    rows = [dict(row) for row in phase8.large_performance_rows]
    rows.append(
        {
            "strategy": "small_dynamic_adaptive_rom",
            "dynamic_dofs": phase8.small_dynamic_rows[0]["rom_max_basis_dimension"],
            "full_free_dofs": phase8.small_dynamic_rows[0]["full_free_dofs"],
            "measured_wall_runtime_ratio": phase8.summary["small_dynamic_measured_wall_runtime_ratio"],
            "measured_wall_speedup": phase8.summary["small_dynamic_measured_wall_speedup"],
            "solver_dimension_speedup": phase8.summary["small_dynamic_solver_dimension_speedup"],
        }
    )
    return rows


def _ablation_rows(phase8: Phase8ValidationMatrix, single_scale: dict[str, float | int]) -> list[Phase9Row]:
    residual_dynamic = int(float(phase8.summary["quasi_static_residual_dynamic_dofs"]))
    ordinary_dynamic = int(float(phase8.summary["quasi_static_ordinary_patch_static_dynamic_dofs"]))
    total_residual = int(phase8.summary["static_total_residual_columns"])
    multi_row = phase8.moving_contact_rows[0]
    return [
        {
            "ablation": "A",
            "model": "only_low_free_interface_modes",
            "quasi_static_force_error": phase8.summary["quasi_static_low_mode_force_error"],
            "static_mean_compliance_error": phase8.summary["static_low_mode_mean_compliance_error"],
            "dynamic_dofs": residual_dynamic,
            "ilc_alpha_dofs": 0,
            "max_active_patch_count": 0,
            "runtime_proxy": float(residual_dynamic**3),
            "claim_role": "negative control for Claim 1",
        },
        {
            "ablation": "B",
            "model": "low_modes_plus_ordinary_patch_static_modes",
            "quasi_static_force_error": phase8.summary["quasi_static_ordinary_patch_static_force_error"],
            "static_mean_compliance_error": 0.0,
            "dynamic_dofs": ordinary_dynamic,
            "ilc_alpha_dofs": 0,
            "max_active_patch_count": total_residual,
            "runtime_proxy": float(ordinary_dynamic**3),
            "claim_role": "accurate but expands the dynamic basis",
        },
        {
            "ablation": "C",
            "model": "low_modes_plus_patch_residual_ilc",
            "quasi_static_force_error": phase8.summary["quasi_static_residual_force_error"],
            "static_mean_compliance_error": phase8.summary["static_residual_max_compliance_error"],
            "dynamic_dofs": residual_dynamic,
            "ilc_alpha_dofs": total_residual,
            "max_active_patch_count": total_residual,
            "runtime_proxy": float(residual_dynamic**3 + total_residual),
            "claim_role": "residual ILC recovers patch static response without entering DAE",
        },
        {
            "ablation": "D",
            "model": "low_modes_plus_sdf_active_patch_residual_ilc",
            "quasi_static_force_error": single_scale["mean_projection_error"],
            "static_mean_compliance_error": phase8.summary["static_residual_max_compliance_error"],
            "dynamic_dofs": residual_dynamic,
            "ilc_alpha_dofs": single_scale["max_active_patch_count"],
            "max_active_patch_count": single_scale["max_active_patch_count"],
            "runtime_proxy": float(residual_dynamic**3 + int(single_scale["max_active_patch_count"]) ** 2),
            "claim_role": "SDF active-set proof for moving contact",
        },
        {
            "ablation": "E",
            "model": "multi_scale_patch_residual_ilc",
            "quasi_static_force_error": phase8.summary["moving_contact_mean_projection_error"],
            "static_mean_compliance_error": phase8.summary["static_residual_max_compliance_error"],
            "dynamic_dofs": residual_dynamic,
            "ilc_alpha_dofs": multi_row["max_active_patch_count"],
            "max_active_patch_count": multi_row["max_active_patch_count"],
            "runtime_proxy": float(residual_dynamic**3 + int(multi_row["max_active_patch_count"]) ** 2),
            "claim_role": "multi-scale accuracy/efficiency tradeoff",
        },
    ]


def _sdf_projection_evidence_rows() -> list[Phase9Row]:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, level.patches)
    right_patches = sorted(
        [patch for patch in level.patches if patch.key[0] == 0 and patch.key[1] == 1],
        key=lambda patch: (patch.centroid[1], patch.centroid[2]),
    )
    if len(right_patches) < 2:
        raise RuntimeError("SDF projection evidence case requires at least two right-side patches")

    case_specs = [
        ("single_patch_probe", right_patches[:1], 1),
        ("two_patch_boundary_probe", right_patches[:2], 2),
        ("outside_surface_no_contact", right_patches[:1], 0),
    ]
    rows: list[Phase9Row] = []
    for case_name, patches, expected_count in case_specs:
        if case_name == "outside_surface_no_contact":
            points = np.asarray([patches[0].centroid + 0.02 * patches[0].normal])
        else:
            points = np.asarray([patch.centroid - 0.02 * patch.normal for patch in patches])
        sample_areas = np.asarray([patch.area for patch in patches], dtype=float)
        samples = detect_sdf_contact_samples(
            points,
            surface,
            level,
            penalty=100.0,
            sample_areas=sample_areas,
        )
        projection = project_contact_samples_to_patch_ilc(fem.mesh, surface, samples, load_bases)
        active_count = int(projection.active_patch_count)
        rows.append(
            {
                "case": case_name,
                "sample_count": int(samples.sample_count),
                "expected_active_patch_count": expected_count,
                "active_patch_count": active_count,
                "active_patch_ids": " ".join(str(patch_id) for patch_id in projection.active_set.active_patch_ids),
                "alpha_dimension": int(projection.active_set.alpha_dimension),
                "dynamic_dimension_contribution": int(projection.active_set.dynamic_dimension_contribution),
                "total_normal_force": float(samples.total_normal_force),
                "projected_alpha_sum": float(np.sum(projection.active_set.alpha)),
                "total_force_error": float(projection.total_force_error),
                "total_normal_force_error": float(projection.total_normal_force_error),
                "total_moment_error": float(projection.total_moment_error),
                "gate_threshold": "active_count=expected; force/normal/moment errors < 1e-12; dynamic contribution = 0",
                "passed": int(
                    active_count == expected_count
                    and float(projection.total_force_error) < 1.0e-12
                    and float(projection.total_normal_force_error) < 1.0e-12
                    and float(projection.total_moment_error) < 1.0e-12
                    and int(projection.active_set.dynamic_dimension_contribution) == 0
                ),
            }
        )
    return rows


def _complex_surface_validation_and_performance_rows() -> tuple[list[Phase9Row], list[Phase9Row]]:
    case = _curved_triangulated_surface_case()
    mesh = case["mesh"]
    surface = case["surface"]
    frames = case["frames"]
    coarse = case["coarse"]
    medium = case["medium"]
    fine_overlap = case["fine_overlap"]
    coarse_bases = case["coarse_bases"]
    medium_bases = case["medium_bases"]
    fine_overlap_bases = case["fine_overlap_bases"]
    penalty = 100.0
    config = AdaptivePatchActivationConfig(
        neighbor_depth=1,
        fine_neighbor_depth=1,
        deactivate_delay=4,
        alpha_blend=0.35,
        max_active_patches=34,
        projection_error_refine_threshold=0.0,
        use_overlap_weights=True,
    )

    def run_active_sequence():
        return run_multiscale_adaptive_patch_sequence(
            mesh,
            surface,
            coarse,
            coarse_bases,
            medium,
            medium_bases,
            fine_overlap,
            fine_overlap_bases,
            frames,
            penalty=penalty,
            config=config,
        )

    active_wall_time, active_result = _median_wall_time(run_active_sequence, repeats=2)
    contact_band = np.asarray(case["contact_triangle_indices"], dtype=np.int64)
    normals = surface.normals[contact_band]
    mean_normal = np.mean(normals, axis=0)
    mean_normal = mean_normal / max(float(np.linalg.norm(mean_normal)), 1.0e-30)
    max_normal_variation_deg = float(np.degrees(np.max(np.arccos(np.clip(normals @ mean_normal, -1.0, 1.0)))))
    unique_requested = len({patch_id for step in active_result.steps for patch_id in step.requested_patch_ids})
    min_gap_history = [float(np.min(step.samples.gaps)) if step.samples.sample_count else 0.0 for step in active_result.steps]
    active_fraction = active_result.max_active_patch_count / max(active_result.total_patch_count, 1)
    complex_row = {
        "case": "warped_triangulated_surface_moving_contact",
        "surface_triangle_count": int(surface.triangles.shape[0]),
        "contact_band_triangle_count": int(contact_band.size),
        "max_contact_normal_variation_deg": max_normal_variation_deg,
        "moving_contact_steps": len(frames),
        "unique_requested_patch_count": int(unique_requested),
        "max_active_patch_count": int(active_result.max_active_patch_count),
        "total_patch_count": int(active_result.total_patch_count),
        "active_patch_fraction": float(active_fraction),
        "mean_projection_error": float(active_result.mean_projection_error),
        "mean_coarse_projection_error": float(active_result.mean_coarse_projection_error),
        "max_force_jump": float(active_result.max_force_jump),
        "max_gap_jump": float(_max_relative_jump(np.asarray(min_gap_history, dtype=float))),
        "optimized_active_wall_time_seconds": float(active_wall_time),
        "gate_threshold": "triangles >= 300; normal variation > 8 deg; unique patches > 4; active fraction < 0.25; projection error < coarse; jumps < 0.2",
        "passed": int(
            int(surface.triangles.shape[0]) >= 300
            and max_normal_variation_deg > 8.0
            and unique_requested > 4
            and active_fraction < 0.25
            and active_result.mean_projection_error < active_result.mean_coarse_projection_error
            and active_result.max_force_jump < 0.2
            and _max_relative_jump(np.asarray(min_gap_history, dtype=float)) < 0.2
        ),
    }

    levels = (coarse, medium, fine_overlap)

    def run_scalar_sdf_kernel() -> tuple[int, float]:
        triangles = surface.nodes[surface.triangles]
        point_count = 0
        checksum = 0.0
        for points, _areas in frames:
            for point in points:
                best_distance = np.inf
                best_point = None
                best_triangle_id = -1
                for triangle_id, triangle in enumerate(triangles):
                    candidate = _closest_point_on_triangle(point, triangle[0], triangle[1], triangle[2])
                    distance = float(np.sum((point - candidate) ** 2))
                    if distance < best_distance:
                        best_distance = distance
                        best_point = candidate
                        best_triangle_id = triangle_id
                assert best_point is not None
                sign = 1.0 if np.dot(point - best_point, surface.normals[best_triangle_id]) >= 0.0 else -1.0
                checksum += sign * float(np.sqrt(best_distance))
                point_count += 1
        return point_count, checksum

    def run_vectorized_sdf_kernel() -> tuple[int, float]:
        point_count = 0
        checksum = 0.0
        for points, _areas in frames:
            query = query_signed_distances(points, surface)
            point_count += int(query.distances.size)
            checksum += float(np.sum(query.distances))
        return point_count, checksum

    def run_uncached_detector() -> int:
        sample_count = 0
        for points, areas in frames:
            for level in levels:
                samples = detect_sdf_contact_samples(points, surface, level, penalty=penalty, sample_areas=areas)
                sample_count += samples.sample_count
        return sample_count

    def run_cached_detector() -> int:
        sample_count = 0
        for points, areas in frames:
            query = query_signed_distances(points, surface)
            for level in levels:
                samples = contact_samples_from_sdf_query(
                    points,
                    surface,
                    level,
                    penalty=penalty,
                    query=query,
                    sample_areas=areas,
                )
                sample_count += samples.sample_count
        return sample_count

    scalar_kernel_wall_time, scalar_kernel_result = _median_wall_time(run_scalar_sdf_kernel, repeats=1)
    vectorized_kernel_wall_time, vectorized_kernel_result = _median_wall_time(run_vectorized_sdf_kernel, repeats=2)
    uncached_wall_time, uncached_samples = _median_wall_time(run_uncached_detector, repeats=2)
    cached_wall_time, cached_samples = _median_wall_time(run_cached_detector, repeats=2)
    kernel_checksum_error = abs(scalar_kernel_result[1] - vectorized_kernel_result[1]) / max(
        abs(scalar_kernel_result[1]),
        1.0e-30,
    )
    kernel_speedup = scalar_kernel_wall_time / max(vectorized_kernel_wall_time, 1.0e-30)
    measured_speedup = uncached_wall_time / max(cached_wall_time, 1.0e-30)
    performance_rows = [
        {
            "case": "exact_vectorized_sdf_kernel",
            "optimization": "batched point-triangle closest-point evaluation with scalar-reference accuracy",
            "scalar_reference_wall_time_seconds": float(scalar_kernel_wall_time),
            "vectorized_wall_time_seconds": float(vectorized_kernel_wall_time),
            "measured_speedup": float(kernel_speedup),
            "sample_count_delta": int(abs(int(scalar_kernel_result[0]) - int(vectorized_kernel_result[0]))),
            "distance_checksum_relative_error": float(kernel_checksum_error),
            "gate_threshold": "speedup > 5; identical point count; distance checksum relative error < 1e-12",
            "passed": int(
                kernel_speedup > 5.0
                and int(scalar_kernel_result[0]) == int(vectorized_kernel_result[0])
                and kernel_checksum_error < 1.0e-12
            ),
        },
        {
            "case": "cached_multiscale_sdf_mapping",
            "optimization": "reuse one vectorized signed-distance query across coarse/medium/fine patch levels",
            "baseline_sdf_query_count": len(frames) * len(levels),
            "optimized_sdf_query_count": len(frames),
            "uncached_wall_time_seconds": float(uncached_wall_time),
            "cached_wall_time_seconds": float(cached_wall_time),
            "measured_speedup": float(measured_speedup),
            "sample_count_delta": int(abs(int(uncached_samples) - int(cached_samples))),
            "gate_threshold": "speedup > 1.4 and identical active sample count",
            "passed": int(measured_speedup > 1.4 and int(uncached_samples) == int(cached_samples)),
        },
        {
            "case": "optimized_active_patch_sequence",
            "optimization": "vectorized SDF mapping plus cached basis maps and vectorized overlap-alpha aggregation",
            "optimized_active_wall_time_seconds": float(active_wall_time),
            "max_active_patch_count": int(active_result.max_active_patch_count),
            "total_patch_count": int(active_result.total_patch_count),
            "active_patch_fraction": float(active_fraction),
            "estimated_active_vs_full_patch_runtime_ratio": float(active_result.mean_runtime_ratio),
            "estimated_active_vs_full_patch_speedup": float(1.0 / max(active_result.mean_runtime_ratio, 1.0e-30)),
            "gate_threshold": "active fraction < 0.25 and runtime ratio < 0.25",
            "passed": int(active_fraction < 0.25 and active_result.mean_runtime_ratio < 0.25),
        },
    ]
    return [complex_row], performance_rows


def _three_way_external_evidence_rows() -> list[Phase9Row]:
    linear_report = _load_three_way_report(
        live_path=_repo_root() / "outputs" / "calculix_three_way" / "three_way_report.json",
        archived_path=_reference_evidence_dir() / "three_way_report.json",
    )
    contact_report = _load_three_way_report(
        live_path=_repo_root() / "outputs" / "calculix_contact_three_way" / "three_way_contact_report.json",
        archived_path=_reference_evidence_dir() / "three_way_contact_report.json",
    )
    rows: list[Phase9Row] = []
    if linear_report is not None:
        payload, source_kind, source_report = linear_report
        full_vs_ccx = payload["python_full_vs_calculix"]
        rom_vs_full = payload["adaptive_rom_vs_python_full"]
        rom_vs_ccx = payload["adaptive_rom_vs_calculix"]
        rows.append(
            {
                "case": "linear_dynamic_three_way",
                "source_kind": source_kind,
                "source_report": str(source_report),
                "external_solver": payload["external_solver"],
                "python_full_vs_calculix_relative_l2": float(full_vs_ccx["relative_observed_displacement_l2"]),
                "adaptive_rom_vs_python_full_relative_l2": float(rom_vs_full["relative_displacement_l2"]),
                "adaptive_rom_vs_calculix_relative_l2": float(rom_vs_ccx["relative_observed_displacement_l2"]),
                "max_observed_displacement_error": float(rom_vs_ccx["max_observed_displacement_error"]),
                "observed_node_count": int(rom_vs_ccx["node_count"]),
                "sample_count": int(rom_vs_ccx["sample_count"]),
                "primary_figure": _live_report_figure(payload, "figure", source_kind),
                "gate_threshold": "all displacement relative L2 < 0.05",
                "passed": int(
                    float(full_vs_ccx["relative_observed_displacement_l2"]) < 5.0e-2
                    and float(rom_vs_full["relative_displacement_l2"]) < 5.0e-2
                    and float(rom_vs_ccx["relative_observed_displacement_l2"]) < 5.0e-2
                ),
            }
        )
    if contact_report is not None:
        payload, source_kind, source_report = contact_report
        matrix = payload["sfc_matrix_vs_calculix_matrix_storage"]
        full_vs_ccx = payload["python_full_vs_calculix"]
        rom_vs_full = payload["adaptive_rom_vs_python_full"]
        rom_vs_ccx = payload["adaptive_rom_vs_calculix"]
        pressure_slope = float(payload["calculix_output_pressure_overclosure_lsq_slope"])
        target_stiffness = float(payload["calculix_contact_stiffness"])
        pressure_slope_error = abs(pressure_slope - target_stiffness) / max(abs(target_stiffness), 1.0e-30)
        rows.append(
            {
                "case": "nonlinear_contact_three_way",
                "source_kind": source_kind,
                "source_report": str(source_report),
                "external_solver": payload["external_solver"],
                "external_open_source_reference": payload["external_open_source_reference"],
                "stiffness_matrix_relative_l2": float(matrix["stiffness_free_dof_relative_l2"]),
                "mass_matrix_relative_l2": float(matrix["mass_free_dof_relative_l2"]),
                "python_full_vs_calculix_relative_l2": float(full_vs_ccx["relative_observed_displacement_l2"]),
                "adaptive_rom_vs_python_full_relative_l2": float(rom_vs_full["relative_displacement_l2"]),
                "adaptive_rom_vs_python_full_force_l2": float(rom_vs_full["relative_force_l2"]),
                "adaptive_rom_vs_calculix_relative_l2": float(rom_vs_ccx["relative_observed_displacement_l2"]),
                "calculix_pressure_overclosure_lsq_slope": pressure_slope,
                "target_pressure_overclosure_stiffness": target_stiffness,
                "pressure_overclosure_slope_relative_error": pressure_slope_error,
                "calculix_contact_stress_rows": int(payload["calculix_contact_stress_rows"]),
                "calculix_pressure_overclosure_sample_count": int(payload["calculix_pressure_overclosure_sample_count"]),
                "max_calculix_contact_pressure": float(payload["max_calculix_contact_pressure"]),
                "max_calculix_contact_overclosure": float(payload["max_calculix_contact_overclosure"]),
                "primary_figure": _live_report_figure(payload, "displacement_figure", source_kind),
                "contact_activity_figure": _live_report_figure(payload, "contact_activity_figure", source_kind),
                "gate_threshold": "matrix rel L2 < 1e-10; displacement relative L2 < 0.02; ROM/full force L2 < 1e-8; pressure slope error < 1e-6",
                "passed": int(
                    float(matrix["stiffness_free_dof_relative_l2"]) < 1.0e-10
                    and float(matrix["mass_free_dof_relative_l2"]) < 1.0e-10
                    and float(full_vs_ccx["relative_observed_displacement_l2"]) < 2.0e-2
                    and float(rom_vs_full["relative_force_l2"]) < 1.0e-8
                    and float(rom_vs_ccx["relative_observed_displacement_l2"]) < 2.0e-2
                    and pressure_slope_error < 1.0e-6
                ),
            }
        )
    return rows


def _claim_rows(
    phase8: Phase8ValidationMatrix,
    single_scale: dict[str, float | int],
    sdf_projection_rows: list[Phase9Row],
    complex_surface_rows: list[Phase9Row],
    performance_rows: list[Phase9Row],
    three_way_external_rows: list[Phase9Row],
) -> list[Phase9Row]:
    summary = phase8.summary
    multi_error = float(summary["moving_contact_mean_projection_error"])
    single_error = float(single_scale["mean_projection_error"])
    return [
        {
            "claim_id": 1,
            "claim": "SDF contact samples map to active patch ILCs with conserved resultant force and zero main-DAE dimension contribution.",
            "passed": int(sdf_projection_rows and all(int(row["passed"]) == 1 for row in sdf_projection_rows)),
            "metric": "sdf_projection_case_pass_count",
            "value": f"{sum(int(row['passed']) for row in sdf_projection_rows)} / {len(sdf_projection_rows)}",
            "threshold": "all SDF projection cases pass",
        },
        {
            "claim_id": 2,
            "claim": "Low-order modes alone cannot represent local surface contact compliance.",
            "passed": int(float(summary["quasi_static_low_mode_force_error"]) > 0.2),
            "metric": "quasi_static_low_mode_force_error",
            "value": summary["quasi_static_low_mode_force_error"],
            "threshold": "> 0.2",
        },
        {
            "claim_id": 3,
            "claim": "Patch residual modes recover local quasistatic response under contact loads.",
            "passed": int(
                float(summary["static_max_completeness_error"]) < 1.0e-6
                and float(summary["quasi_static_residual_force_error"]) < 1.0e-8
            ),
            "metric": "static_completeness_and_residual_force_error",
            "value": (
                f"{summary['static_max_completeness_error']:.6g};"
                f"{summary['quasi_static_residual_force_error']:.6g}"
            ),
            "threshold": "< 1e-6; < 1e-8",
        },
        {
            "claim_id": 4,
            "claim": "Patch-level ILC reduces the scale versus node-level ILC.",
            "passed": int(int(summary["large_multiscale_patch_alpha_dofs"]) < int(summary["large_node_level_contact_dofs"])),
            "metric": "patch_alpha_dofs_vs_node_level_contact_dofs",
            "value": f"{summary['large_multiscale_patch_alpha_dofs']} < {summary['large_node_level_contact_dofs']}",
            "threshold": "patch < node",
        },
        {
            "claim_id": 5,
            "claim": "Active patch ILC alpha does not enter the main DAE state as the active contact set changes.",
            "passed": int(
                int(summary["large_dae_dimension_is_contact_independent"]) == 1
                and float(summary["quasi_static_residual_dynamic_dof_ratio_vs_ordinary"]) < 1.0
            ),
            "metric": "dae_dimension_independence_and_dynamic_dof_ratio",
            "value": (
                f"{summary['large_dae_dimension_is_contact_independent']};"
                f"{summary['quasi_static_residual_dynamic_dof_ratio_vs_ordinary']:.6g}"
            ),
            "threshold": "1; < 1",
        },
        {
            "claim_id": 6,
            "claim": "Multi-scale overlapping patch activation handles moving contact over patch boundaries, including a warped triangular surface case.",
            "passed": int(
                float(summary["moving_contact_force_jump"]) < 5.0e-2
                and float(summary["moving_contact_gap_jump"]) < 5.0e-2
                and int(single_scale["unique_requested_patch_count"]) > 1
                and multi_error <= single_error
                and float(summary["large_active_patch_fraction"]) < 0.35
                and float(summary["large_runtime_ratio_active_vs_full_patch"]) < 0.35
                and len(complex_surface_rows) > 0
                and all(int(row["passed"]) == 1 for row in complex_surface_rows)
            ),
            "metric": "force_jump_gap_jump_unique_requested_patches_multi_error_single_error_active_fraction_runtime_ratio_complex_cases",
            "value": (
                f"{summary['moving_contact_force_jump']:.6g};"
                f"{summary['moving_contact_gap_jump']:.6g};"
                f"{single_scale['unique_requested_patch_count']};"
                f"{multi_error:.6g};"
                f"{single_error:.6g};"
                f"{summary['large_active_patch_fraction']:.6g};"
                f"{summary['large_runtime_ratio_active_vs_full_patch']:.6g};"
                f"{sum(int(row['passed']) for row in complex_surface_rows)} / {len(complex_surface_rows)}"
            ),
            "threshold": "< 0.05; < 0.05; > 1; multi <= single; active/runtime < 0.35; all complex surface cases pass",
        },
        {
            "claim_id": 7,
            "claim": "Three-way FEM evidence shows the result is not only Python full-FEM/ROM self-consistency.",
            "passed": int(three_way_external_rows and all(int(row["passed"]) == 1 for row in three_way_external_rows)),
            "metric": "three_way_external_case_pass_count",
            "value": f"{sum(int(row['passed']) for row in three_way_external_rows)} / {len(three_way_external_rows)}",
            "threshold": "linear and nonlinear external FEM evidence cases pass",
        },
        {
            "claim_id": 8,
            "claim": "The implementation now has measured performance evidence from cached SDF mapping and active patch evaluation.",
            "passed": int(len(performance_rows) > 0 and all(int(row["passed"]) == 1 for row in performance_rows)),
            "metric": "performance_optimization_case_pass_count",
            "value": f"{sum(int(row['passed']) for row in performance_rows)} / {len(performance_rows)}",
            "threshold": "all performance optimization evidence cases pass",
        },
    ]


def _summary(
    phase8: Phase8ValidationMatrix,
    single_scale: dict[str, float | int],
    sdf_projection_rows: list[Phase9Row],
    complex_surface_rows: list[Phase9Row],
    performance_rows: list[Phase9Row],
    three_way_external_rows: list[Phase9Row],
    claim_rows: list[Phase9Row],
) -> dict[str, float | int | str]:
    passed = int(all(int(row["passed"]) == 1 for row in claim_rows))
    return {
        "phase8_gate_passed": int(phase8.gate_passed),
        "phase9_claims_passed": int(sum(int(row["passed"]) for row in claim_rows)),
        "phase9_claim_count": len(claim_rows),
        "phase9_gate_passed": passed,
        "single_scale_projection_error": float(single_scale["mean_projection_error"]),
        "multi_scale_projection_error": float(phase8.summary["moving_contact_mean_projection_error"]),
        "node_level_contact_dofs": int(phase8.summary["large_node_level_contact_dofs"]),
        "multi_scale_patch_alpha_dofs": int(phase8.summary["large_multiscale_patch_alpha_dofs"]),
        "sdf_projection_cases": len(sdf_projection_rows),
        "sdf_projection_cases_passed": int(sum(int(row["passed"]) for row in sdf_projection_rows)),
        "complex_surface_cases": len(complex_surface_rows),
        "complex_surface_cases_passed": int(sum(int(row["passed"]) for row in complex_surface_rows)),
        "performance_optimization_cases": len(performance_rows),
        "performance_optimization_cases_passed": int(sum(int(row["passed"]) for row in performance_rows)),
        "three_way_external_cases": len(three_way_external_rows),
        "three_way_external_cases_passed": int(sum(int(row["passed"]) for row in three_way_external_rows)),
    }


def _write_claims_markdown(path: Path, result: Phase9ClaimGate) -> None:
    lines = [
        "# Patch ILC Claim Gate\n\n",
        f"Phase 9 gate passed: {int(result.gate_passed)}\n",
        f"Claims passed: {result.summary['phase9_claims_passed']} / {result.summary['phase9_claim_count']}\n\n",
        "## Ablation Set\n\n",
    ]
    for row in result.ablation_rows:
        lines.append(
            f"- {row['ablation']} {row['model']}: force/error={float(row['quasi_static_force_error']):.6g}, "
            f"dynamic_dofs={row['dynamic_dofs']}, alpha_or_patch_dofs={row['ilc_alpha_dofs']}\n"
        )
    lines.append("\n## Claims\n\n")
    for row in result.claim_rows:
        status = "PASS" if int(row["passed"]) else "FAIL"
        lines.append(
            f"- Claim {row['claim_id']} [{status}]: {row['claim']} "
            f"Metric {row['metric']}={row['value']} threshold {row['threshold']}.\n"
        )
    lines.extend(
        [
            "\n## Evidence Tables\n\n",
            "- `sdf_patch_projection.csv`: SDF sample to active patch ILC mapping, force conservation, and zero dynamic-state contribution.\n",
            "- `complex_surface_moving_contact.csv`: warped triangular surface moving-contact validation.\n",
            "- `performance_optimization.csv`: cached SDF mapping and active patch performance evidence.\n",
            "- `three_way_external_fem.csv`: Python full FEM, adaptive ROM, and CalculiX external FEM comparison.\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def _write_phase9_figures(result: Phase9ClaimGate, figures: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_method_pipeline(figures / "method_pipeline.png", plt)
    _plot_patch_hierarchy(figures / "patch_hierarchy.png", plt)
    _plot_force_displacement(figures / "force_displacement_curve.png", result, plt)
    _plot_active_patch_history(figures / "active_patch_history.png", result, plt)
    _plot_runtime_scaling(figures / "runtime_scaling.png", result, plt)
    _plot_sdf_projection_evidence(figures / "sdf_patch_projection_map.png", result, plt)
    _plot_complex_surface_case(figures / "complex_surface_moving_contact.png", plt)
    _plot_performance_optimization(figures / "performance_optimization.png", result, plt)


def _plot_method_pipeline(path: Path, plt) -> None:
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(11.2, 5.2), constrained_layout=True)
    ax.axis("off")
    panels = [
        (0.03, 0.56, 0.15, 0.34, "a  full FEM\nmoving contact"),
        (0.22, 0.56, 0.15, 0.34, "b  SDF query\ntriangle/gap/normal"),
        (0.41, 0.56, 0.15, 0.34, "c  patch load\narea weighting"),
        (0.60, 0.56, 0.15, 0.34, "d  residual ILC\nlocal compliance"),
        (0.79, 0.56, 0.18, 0.34, "e  active alpha\noutside main DAE"),
    ]
    for x, y, width, height, label in panels:
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=1.1,
                edgecolor="#333333",
                facecolor="#f7f7f7",
                transform=ax.transAxes,
            )
        )
        ax.text(x + 0.012, y + height - 0.055, label, ha="left", va="top", fontsize=9.2, transform=ax.transAxes)

    ax.add_patch(patches.Rectangle((0.055, 0.63), 0.09, 0.10, angle=-8, color="#b9c7d8", transform=ax.transAxes))
    ax.plot([0.07, 0.10, 0.13], [0.80, 0.83, 0.79], color="#d55e00", linewidth=2, transform=ax.transAxes)
    ax.scatter([0.10], [0.83], s=45, color="#d55e00", transform=ax.transAxes)

    tri = patches.Polygon([[0.255, 0.65], [0.335, 0.65], [0.295, 0.78]], closed=True, fill=False, edgecolor="#4c78a8", transform=ax.transAxes)
    ax.add_patch(tri)
    ax.annotate("", xy=(0.298, 0.70), xytext=(0.318, 0.83), arrowprops={"arrowstyle": "->", "color": "#d55e00"}, xycoords=ax.transAxes)
    ax.text(0.303, 0.78, "gap", fontsize=8, color="#d55e00", transform=ax.transAxes)

    for i in range(4):
        for j in range(3):
            ax.add_patch(
                patches.Rectangle(
                    (0.435 + 0.025 * i, 0.64 + 0.045 * j),
                    0.023,
                    0.040,
                    linewidth=0.6,
                    edgecolor="#666666",
                    facecolor="#dcecc9" if i in (1, 2) else "#ffffff",
                    transform=ax.transAxes,
                )
            )
    ax.annotate("sum Fn", xy=(0.49, 0.75), xytext=(0.50, 0.64), fontsize=7.5, arrowprops={"arrowstyle": "->"}, xycoords=ax.transAxes, textcoords=ax.transAxes)

    xs = np.linspace(0.63, 0.72, 80)
    ax.plot(xs, 0.69 + 0.04 * np.sin(80 * xs), color="#4c78a8", linewidth=1.8, transform=ax.transAxes)
    ax.plot(xs, 0.73 + 0.02 * np.sin(105 * xs), color="#f28e2b", linewidth=1.5, transform=ax.transAxes)
    ax.text(0.635, 0.61, "static attachment\n- modal part", fontsize=7.5, transform=ax.transAxes)

    ax.add_patch(patches.Rectangle((0.815, 0.66), 0.07, 0.08, linewidth=1.0, edgecolor="#4c78a8", facecolor="#e8f1fb", transform=ax.transAxes))
    ax.add_patch(patches.Rectangle((0.895, 0.66), 0.045, 0.08, linewidth=1.0, edgecolor="#d55e00", facecolor="#fde8d6", transform=ax.transAxes))
    ax.text(0.850, 0.70, "q", ha="center", va="center", fontsize=12, transform=ax.transAxes)
    ax.text(0.918, 0.70, "alpha", ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.annotate("", xy=(0.895, 0.70), xytext=(0.885, 0.70), arrowprops={"arrowstyle": "->"}, xycoords=ax.transAxes)

    for (x0, _y0, width0, _h0, _label0), (x1, _y1, _width1, _h1, _label1) in zip(panels[:-1], panels[1:]):
        ax.annotate(
            "",
            xy=(x1 - 0.012, 0.73),
            xytext=(x0 + width0 + 0.012, 0.73),
            arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#333333"},
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )

    ladder = [
        "static\ncompleteness",
        "SDF\nprojection",
        "quasi-static\ncontact",
        "moving\ncontact",
        "warped\nsurface",
        "CalculiX\nthree-way",
        "performance\nscaling",
    ]
    ladder_xs = np.linspace(0.08, 0.92, len(ladder))
    for index, (x, label) in enumerate(zip(ladder_xs, ladder)):
        ax.scatter([x], [0.28], s=145, color="#59a14f", edgecolor="#333333", linewidth=0.7, transform=ax.transAxes, zorder=3)
        ax.text(x, 0.28, str(index + 1), ha="center", va="center", color="white", fontsize=9, transform=ax.transAxes, zorder=4)
        ax.text(x, 0.18, label, ha="center", va="top", fontsize=8.2, transform=ax.transAxes)
    ax.plot(ladder_xs, [0.28] * len(ladder_xs), color="#333333", linewidth=1.0, transform=ax.transAxes, zorder=2)
    ax.text(0.03, 0.36, "validation ladder", ha="left", va="center", fontsize=9.5, fontweight="bold", transform=ax.transAxes)
    ax.set_title("SDF-driven patch residual ILC for moving surface contact")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_patch_hierarchy(path: Path, plt) -> None:
    fem = cantilever_block(nx=2, ny=8, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(4, 2), fine_bins=(8, 4))
    names = [level.name for level in hierarchy.levels]
    counts = [level.patch_count for level in hierarchy.levels]
    fig, ax = plt.subplots(figsize=(8.5, 4.0), constrained_layout=True)
    colors = ["#4c78a8", "#59a14f", "#8cd17d", "#f28e2b", "#ffbe7d"][: len(names)]
    ax.bar(names, counts, color=colors)
    ax.set_ylabel("patch count")
    ax.set_title("Coarse / medium / fine patch hierarchy")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_force_displacement(path: Path, result: Phase9ClaimGate, plt) -> None:
    rows = [row for row in result.phase8.quasi_static_rows if row["patch_id"] == result.phase8.quasi_static_rows[0]["patch_id"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for model in sorted({row["model"] for row in rows}):
        model_rows = sorted((row for row in rows if row["model"] == model), key=lambda item: item["indentation"])
        ax.plot(
            [row["indentation"] for row in model_rows],
            [row["predicted_force"] for row in model_rows],
            marker="o",
            label=model,
        )
    ax.set_xlabel("indentation")
    ax.set_ylabel("normal force")
    ax.set_title("Ablation force-displacement response")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_active_patch_history(path: Path, result: Phase9ClaimGate, plt) -> None:
    history = sorted(result.active_patch_history_rows, key=lambda row: row["step"])
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.4), sharex=True, constrained_layout=True)
    axes[0].plot([row["step"] for row in history], [row["active_patch_count"] for row in history], color="#4c78a8")
    axes[0].set_ylabel("active patches")
    axes[0].grid(True, alpha=0.35)
    axes[1].plot([row["step"] for row in history], [row["projected_normal_force"] for row in history], color="#f28e2b")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("normal force")
    axes[1].grid(True, alpha=0.35)
    fig.suptitle("SDF-driven active patch history")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_runtime_scaling(path: Path, result: Phase9ClaimGate, plt) -> None:
    rows = [row for row in result.runtime_rows if "runtime_proxy" in row]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    labels = [str(row["strategy"]) for row in rows]
    proxies = [float(row["runtime_proxy"]) for row in rows]
    ax.bar(labels, proxies, color="#4c78a8")
    ax.set_ylabel("runtime proxy")
    ax.set_title("Runtime scaling by strategy")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_sdf_projection_evidence(path: Path, result: Phase9ClaimGate, plt) -> None:
    rows = list(result.sdf_projection_rows)
    labels = [str(row["case"]).replace("_", "\n") for row in rows]
    active_counts = [int(row["active_patch_count"]) for row in rows]
    expected_counts = [int(row["expected_active_patch_count"]) for row in rows]
    normal_errors = [float(row["total_normal_force_error"]) for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True, constrained_layout=True)
    axes[0].bar(x - 0.18, expected_counts, width=0.36, label="expected", color="#4c78a8")
    axes[0].bar(x + 0.18, active_counts, width=0.36, label="projected", color="#59a14f")
    axes[0].set_ylabel("active patches")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, normal_errors, color="#f28e2b")
    axes[1].set_ylabel("normal force rel. error")
    axes[1].set_xticks(x, labels)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("SDF contact sample to active patch ILC evidence")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_complex_surface_case(path: Path, plt) -> None:
    case = _curved_triangulated_surface_case()
    surface = case["surface"]
    frames = case["frames"]
    contact_path = []
    for points, _areas in frames:
        if points.size:
            contact_path.append(np.mean(points, axis=0))
    path_points = np.asarray(contact_path, dtype=float)
    fig = plt.figure(figsize=(8.6, 5.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        surface.nodes[:, 0],
        surface.nodes[:, 1],
        surface.nodes[:, 2],
        triangles=surface.triangles,
        color="#b9c7d8",
        alpha=0.34,
        linewidth=0.2,
        edgecolor="#6b7280",
    )
    if path_points.size:
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color="#d55e00", linewidth=2.2)
        ax.scatter(path_points[[0, -1], 0], path_points[[0, -1], 1], path_points[[0, -1], 2], color="#0072b2", s=26)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Warped triangular surface moving contact case")
    ax.view_init(elev=18, azim=-60)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_performance_optimization(path: Path, result: Phase9ClaimGate, plt) -> None:
    rows = list(result.performance_optimization_rows)
    kernel = next((row for row in rows if row["case"] == "exact_vectorized_sdf_kernel"), None)
    detector = next((row for row in rows if row["case"] == "cached_multiscale_sdf_mapping"), None)
    active = next((row for row in rows if row["case"] == "optimized_active_patch_sequence"), None)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    if kernel is not None:
        axes[0].bar(
            ["scalar\nquery", "batched\nquery"],
            [float(kernel["scalar_reference_wall_time_seconds"]), float(kernel["vectorized_wall_time_seconds"])],
            color=["#4c78a8", "#59a14f"],
        )
        axes[0].set_ylabel("wall time (s)")
        axes[0].set_title(f"exact SDF kernel speedup {float(kernel['measured_speedup']):.1f}x")
    elif detector is not None:
        axes[0].bar(
            ["uncached", "cached"],
            [float(detector["uncached_wall_time_seconds"]), float(detector["cached_wall_time_seconds"])],
            color=["#4c78a8", "#59a14f"],
        )
        axes[0].set_ylabel("wall time (s)")
        axes[0].set_title(f"SDF mapping speedup {float(detector['measured_speedup']):.2f}x")
    if active is not None:
        active_count = float(active["max_active_patch_count"])
        inactive_count = float(active["total_patch_count"]) - active_count
        axes[1].bar(
            ["patches"],
            [active_count],
            color="#f28e2b",
            label="active",
        )
        axes[1].bar(
            ["patches"],
            [inactive_count],
            bottom=[active_count],
            color="#d9d9d9",
            label="inactive",
        )
        axes[1].set_ylabel("patch count")
        axes[1].set_title(
            f"active fraction {float(active['active_patch_fraction']):.2f}, "
            f"proxy speedup {float(active['estimated_active_vs_full_patch_speedup']):.1f}x"
        )
        axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _copy_three_way_figures(rows: tuple[Phase9Row, ...], figures: Path) -> None:
    figure_map = {
        "linear_dynamic_three_way": "three_way_dynamic_displacement.png",
        "nonlinear_contact_three_way": "three_way_nonlinear_contact_displacement.png",
    }
    for row in rows:
        case = str(row.get("case", ""))
        source = Path(str(row.get("primary_figure", "")))
        target_name = figure_map.get(case)
        if target_name and source.is_file():
            shutil.copyfile(source, figures / target_name)
        contact_source = Path(str(row.get("contact_activity_figure", "")))
        if case == "nonlinear_contact_three_way" and contact_source.is_file():
            shutil.copyfile(contact_source, figures / "three_way_nonlinear_contact_activity.png")


def _write_csv(path: Path, rows: tuple[Phase9Row, ...] | list[Phase9Row]) -> None:
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


def _load_three_way_report(
    live_path: Path,
    archived_path: Path,
) -> tuple[dict[str, Any], str, Path] | None:
    if live_path.is_file():
        return json.loads(live_path.read_text(encoding="utf-8")), "live_outputs", live_path
    if archived_path.is_file():
        return json.loads(archived_path.read_text(encoding="utf-8")), "archived_external_evidence", archived_path
    return None


def _live_report_figure(payload: dict[str, Any], key: str, source_kind: str) -> str:
    if source_kind != "live_outputs":
        return ""
    return str(payload.get(key, ""))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _reference_evidence_dir() -> Path:
    return Path(__file__).resolve().with_name("reference_evidence")


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


def _curved_triangulated_surface_case() -> dict[str, Any]:
    base = cantilever_block(nx=4, ny=10, nz=3, length=4.0, width=1.2, height=1.0, stiffness_scale=100.0)
    nodes = base.mesh.nodes.copy()
    x_max = float(nodes[:, 0].max())
    y_min, y_max = float(nodes[:, 1].min()), float(nodes[:, 1].max())
    z_min, z_max = float(nodes[:, 2].min()), float(nodes[:, 2].max())
    xi = nodes[:, 0] / max(x_max, 1.0e-30)
    eta = (nodes[:, 1] - y_min) / max(y_max - y_min, 1.0e-30)
    zeta = (nodes[:, 2] - z_min) / max(z_max - z_min, 1.0e-30)
    right_weight = xi**2
    nodes[:, 0] += right_weight * 0.22 * np.sin(2.0 * np.pi * eta) * np.cos(np.pi * zeta)
    nodes[:, 1] += right_weight * 0.04 * np.sin(np.pi * xi) * np.sin(2.0 * np.pi * zeta)
    nodes[:, 2] += right_weight * 0.07 * np.sin(np.pi * eta) * np.sin(2.0 * np.pi * zeta)
    mesh = Mesh(nodes, base.mesh.elements, base.mesh.element_type)
    surface = extract_surface(mesh)
    hierarchy = build_patch_hierarchy(
        surface,
        coarse_bins=(1, 1),
        medium_bins=(5, 3),
        fine_bins=(8, 4),
    )
    coarse = hierarchy.level("coarse")
    medium = hierarchy.level("medium")
    fine_overlap = hierarchy.level("fine_overlap")
    frames, contact_triangle_indices = _curved_surface_sliding_frames(surface, steps=35)
    return {
        "mesh": mesh,
        "surface": surface,
        "coarse": coarse,
        "medium": medium,
        "fine_overlap": fine_overlap,
        "coarse_bases": build_normal_patch_load_bases(mesh, surface, coarse.patches),
        "medium_bases": build_normal_patch_load_bases(mesh, surface, medium.patches),
        "fine_overlap_bases": build_normal_patch_load_bases(mesh, surface, fine_overlap.patches),
        "frames": frames,
        "contact_triangle_indices": contact_triangle_indices,
    }


def _curved_surface_sliding_frames(surface: Any, steps: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    x_cut = float(np.quantile(surface.centroids[:, 0], 0.82))
    contact_triangles = np.flatnonzero((surface.normals[:, 0] > 0.45) & (surface.centroids[:, 0] >= x_cut))
    if contact_triangles.size < 10:
        contact_triangles = np.flatnonzero(surface.normals[:, 0] > 0.30)
    if contact_triangles.size == 0:
        raise RuntimeError("curved surface moving contact case needs a positive-x contact band")
    yz = surface.centroids[contact_triangles][:, 1:3]
    y_min, y_max = float(yz[:, 0].min()), float(yz[:, 0].max())
    z_min, z_max = float(yz[:, 1].min()), float(yz[:, 1].max())
    y_span = max(y_max - y_min, 1.0e-30)
    z_span = max(z_max - z_min, 1.0e-30)
    centers = np.linspace(y_min + 0.15 * y_span, y_max - 0.15 * y_span, steps)
    radius_y = 0.28 * y_span
    radius_z = 0.45 * z_span
    base_penetration = 0.012
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for step_id, y0 in enumerate(centers):
        z0 = 0.18 * z_span * np.sin(2.0 * np.pi * step_id / max(steps - 1, 1))
        points = []
        areas = []
        for triangle_id in contact_triangles:
            centroid = surface.centroids[int(triangle_id)]
            dy = (centroid[1] - y0) / max(radius_y, 1.0e-30)
            dz = (centroid[2] - z0) / max(radius_z, 1.0e-30)
            weight = max(0.0, 1.0 - dy * dy - dz * dz)
            if weight <= 0.0:
                continue
            points.append(centroid - base_penetration * weight * surface.normals[int(triangle_id)])
            areas.append(surface.areas[int(triangle_id)])
        if points:
            frame_points = np.asarray(points, dtype=float)
            frame_areas = np.asarray(areas, dtype=float)
        else:
            frame_points = np.zeros((0, 3), dtype=float)
            frame_areas = np.zeros(0, dtype=float)
        frames.append((frame_points, frame_areas))
    return frames, contact_triangles


def _median_wall_time(callable_obj, repeats: int = 2) -> tuple[float, Any]:
    durations: list[float] = []
    payload: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        payload = callable_obj()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations)), payload


def _max_relative_jump(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    jumps = []
    for previous, current in zip(values[:-1], values[1:]):
        scale = max(abs(float(previous)), abs(float(current)), 1.0e-30)
        jumps.append(abs(float(current) - float(previous)) / scale)
    return float(max(jumps, default=0.0))
