from __future__ import annotations

from dataclasses import asdict
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.contact_dynamics import (
    AdaptiveCalculixAlignedROMContactSimulator,
    CalculixAlignedFullFEMContactSimulator,
    PrescribedRigidPlane,
    patch_mode_dict,
)
from modal_contact_rom.contact_modes import build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import (
    CalculixNodeHistory,
    cantilever_block,
    matrix_storage_to_fem_system,
    read_matrix_storage,
    read_node_print_displacements,
    run_ccx_wsl,
    sfc_cantilever_block,
    write_cantilever_matrix_storage_input,
    write_cantilever_plane_contact_dynamic_input,
)
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.reduced_dynamics import compute_craig_bampton_basis, mass_orthonormalize
from modal_contact_rom.surface_patch import extract_surface, partition_surface_patches
from modal_contact_rom.validation import (
    CalculixObservedDisplacementSummary,
    compare_dynamic_results,
)


def main() -> None:
    report = run_three_way_contact_benchmark()
    print(json.dumps(report, indent=2))


def run_three_way_contact_benchmark() -> dict[str, object]:
    workdir = ROOT / "outputs" / "calculix_contact_three_way"
    workdir.mkdir(parents=True, exist_ok=True)
    matrix_job = "contact_three_way_mtx"
    contact_job = "contact_three_way_nonlinear"
    young_modulus = 1000.0
    poisson_ratio = 0.3
    density = 1.0
    dt = 0.000625
    steps = 281
    plane_z = 0.525
    calculix_contact_stiffness = 500.0
    upward_force_per_free_top_node = 1.35
    load_ramp_time = 0.08
    rayleigh_alpha = 0.0
    rayleigh_beta = 0.0

    seed = cantilever_block(nx=2, ny=1, nz=1, length=2.0, width=1.0, height=1.0)
    write_cantilever_matrix_storage_input(
        workdir / f"{matrix_job}.inp",
        seed.mesh,
        seed.fixed_nodes,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
    )
    run_ccx_wsl(matrix_job, workdir)
    matrices = read_matrix_storage(matrix_job, workdir)
    calculix_matrix_fem = matrix_storage_to_fem_system(seed.mesh, matrices)
    fem = sfc_cantilever_block(
        nx=2,
        ny=1,
        nz=1,
        length=2.0,
        width=1.0,
        height=1.0,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        mass_kind="consistent",
    )
    matrix_alignment = compare_fem_matrices(fem, calculix_matrix_fem)

    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(1, 1))
    top_patch = next(patch for patch in patches if patch.key[:2] == (2, 1))
    top_nodes = np.flatnonzero(np.isclose(fem.mesh.nodes[:, 2], fem.mesh.nodes[:, 2].max()))
    free_top_nodes = np.setdiff1d(top_nodes, fem.fixed_nodes, assume_unique=False)
    cload = np.zeros(fem.mesh.n_dofs, dtype=float)
    cload[3 * free_top_nodes + 2] = upward_force_per_free_top_node

    write_cantilever_plane_contact_dynamic_input(
        workdir / f"{contact_job}.inp",
        seed.mesh,
        seed.fixed_nodes,
        top_nodes,
        cload,
        plane_z=plane_z,
        time_step=dt,
        total_time=dt * steps,
        contact_stiffness=calculix_contact_stiffness,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        max_increments=steps + 20,
        load_ramp_time=load_ramp_time,
        direct=True,
        contact_type="surface_to_surface",
    )
    run_ccx_wsl(contact_job, workdir, timeout=180)
    calculix_history = read_node_print_displacements(workdir / f"{contact_job}.dat")
    calculix_contact = read_calculix_contact_blocks(workdir / f"{contact_job}.dat")

    rigid_plane = PrescribedRigidPlane(
        point=np.array([0.0, 0.0, plane_z]),
        normal=np.array([0.0, 0.0, 1.0]),
        area_weighted=True,
    )
    nodal_contact_areas = rigid_plane.nodal_area_scales(fem.mesh, surface)
    external_force = lambda time: cload * min(max(time / load_ramp_time, 0.0), 1.0)
    python_contact_penalty = calculix_contact_stiffness
    python_contact_damping = 0.0
    full = CalculixAlignedFullFEMContactSimulator(
        fem=fem,
        rigid_plane=rigid_plane,
        contact_stiffness=python_contact_penalty,
        external_force=external_force,
        hht_alpha=0.0,
    ).run(dt=dt, steps=steps)
    full_vs_calculix = compare_to_calculix_observed_displacements_at_times(full, dt, calculix_history, top_nodes)

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
    residual_activation_count = max(int(contact_modes.patch_ids.size) - 1, 0)
    rom = AdaptiveCalculixAlignedROMContactSimulator(
        fem=fem,
        surface=surface,
        patches=patches,
        base_basis=base_basis,
        patch_modes=patch_mode_dict(contact_modes.patch_ids, contact_modes.modes),
        rigid_plane=rigid_plane,
        contact_stiffness=python_contact_penalty,
        rayleigh_alpha=rayleigh_alpha,
        rayleigh_beta=rayleigh_beta,
        external_force=external_force,
        hht_alpha=0.0,
        activation_count=1,
        residual_activation_count=residual_activation_count,
        retain_active_patches=True,
        max_activation_iterations=max(4, int(contact_modes.patch_ids.size) + 1),
    ).run(dt=dt, steps=steps)

    rom_vs_full = compare_dynamic_results(full, rom)
    rom_vs_calculix = compare_to_calculix_observed_displacements_at_times(rom, dt, calculix_history, top_nodes)

    full_order_figure = workdir / "full_order_external_alignment.png"
    displacement_figure = workdir / "nonlinear_contact_three_way_displacement.png"
    displacement_error_figure = workdir / "nonlinear_contact_three_way_displacement_error.png"
    contact_figure = workdir / "nonlinear_contact_three_way_contact_activity.png"
    representative_node = int(free_top_nodes[-1])
    plot_full_order_alignment(full_order_figure, representative_node, dt, full, calculix_history)
    plot_three_way_top_displacement(displacement_figure, representative_node, dt, full, rom, calculix_history)
    plot_three_way_top_displacement_error(displacement_error_figure, representative_node, dt, full, rom, calculix_history)
    plot_contact_activity(contact_figure, top_nodes, fem.mesh.nodes, plane_z, dt, full, rom, calculix_history)
    representative_node_error = representative_node_displacement_error_summary(
        representative_node,
        dt,
        full,
        rom,
        calculix_history,
    )

    report = {
        "external_solver": "CalculiX ccx via WSL, *DYNAMIC DIRECT with *CONTACT PAIR",
        "external_open_source_reference": "https://github.com/Dhondtguido/CalculiX",
        "matrix_storage_input": str(workdir / f"{matrix_job}.inp"),
        "nonlinear_contact_input": str(workdir / f"{contact_job}.inp"),
        "calculix_contact_type": "surface-to-surface",
        "python_fem_backend": "vendored sfc package",
        "sfc_matrix_vs_calculix_matrix_storage": matrix_alignment,
        "plane_z": plane_z,
        "top_patch_id": int(top_patch.patch_id),
        "top_patch_key": list(top_patch.key),
        "top_nodes": [int(node_id) for node_id in top_nodes],
        "free_top_nodes": [int(node_id) for node_id in free_top_nodes],
        "calculix_contact_stiffness": calculix_contact_stiffness,
        "python_contact_penalty": python_contact_penalty,
        "python_contact_damping": python_contact_damping,
        "python_contact_model": "surface quadrature pressure-overclosure with Newton contact tangent",
        "python_full_order_solver": "SFC-assembled CalculiX-aligned implicit full-order FEM contact",
        "rom_solver": "SFC-assembled adaptive ROM with contact-point and residual patch-mode activation",
        "python_contact_penalty_source": "same pressure-overclosure stiffness as the CalculiX surface-to-surface input",
        "top_contact_area_sum": float(np.sum(nodal_contact_areas[top_nodes])),
        "free_top_contact_area_sum": float(np.sum(nodal_contact_areas[free_top_nodes])),
        "upward_force_per_free_top_node": upward_force_per_free_top_node,
        "load_ramp_time": load_ramp_time,
        "adaptive_base_modes": int(base_basis.shape[1]),
        "adaptive_min_modes": int(min(rom.basis_size_history)),
        "adaptive_max_modes": int(max(rom.basis_size_history)),
        "adaptive_activation_count": 1,
        "adaptive_residual_activation_count": int(residual_activation_count),
        "adaptive_active_patch_ids": [int(patch_id) for patch_id in sorted({pid for active in rom.active_patch_history for pid in active})],
        "full_free_dofs": int(fem.free_dofs.size),
        "max_python_full_contact_force": float(max(step.normal_force for step in full.steps)),
        "max_adaptive_rom_contact_force": float(max(step.normal_force for step in rom.steps)),
        "max_python_full_penetration": float(max(step.max_penetration for step in full.steps)),
        "max_adaptive_rom_penetration": float(max(step.max_penetration for step in rom.steps)),
        "max_calculix_observed_penetration": max_calculix_observed_penetration(calculix_history, top_nodes, seed.mesh.nodes, plane_z),
        "calculix_contact_stress_blocks": int(calculix_contact["block_count"]),
        "calculix_contact_stress_rows": int(calculix_contact["row_count"]),
        "max_calculix_contact_pressure": float(calculix_contact["max_pressure"]),
        "max_calculix_contact_overclosure": float(calculix_contact["max_overclosure"]),
        "calculix_pressure_overclosure_sample_count": int(calculix_contact["sample_count"]),
        "calculix_output_pressure_overclosure_lsq_slope": float(calculix_contact["equivalent_pressure_stiffness"]),
        "calculix_output_max_pressure_overclosure_ratio": float(calculix_contact["max_pressure_overclosure_ratio"]),
        "python_full_vs_calculix": asdict(full_vs_calculix),
        "adaptive_rom_vs_python_full": asdict(rom_vs_full),
        "adaptive_rom_vs_calculix": asdict(rom_vs_calculix),
        "representative_node_displacement_error": representative_node_error,
        "full_order_alignment_figure": str(full_order_figure),
        "displacement_figure": str(displacement_figure),
        "displacement_error_figure": str(displacement_error_figure),
        "contact_activity_figure": str(contact_figure),
    }
    (workdir / "three_way_contact_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_fem_matrices(fem, reference_fem) -> dict[str, float]:
    free = fem.free_dofs
    return {
        "stiffness_free_dof_relative_l2": relative_l2(
            reference_fem.K[free, :][:, free].toarray(),
            fem.K[free, :][:, free].toarray(),
        ),
        "mass_free_dof_relative_l2": relative_l2(
            reference_fem.M[free, :][:, free].toarray(),
            fem.M[free, :][:, free].toarray(),
        ),
    }


def plot_full_order_alignment(
    path: Path,
    node_id: int,
    dt: float,
    full,
    calculix_history: CalculixNodeHistory,
) -> None:
    time = dt * (np.arange(full.displacement_history.shape[0]) + 1)
    py = full.displacement_history[:, 3 * node_id + 2]
    ccx = calculix_history.displacements[node_id][:, 2]
    ccx_time = calculix_history.times[: ccx.shape[0]]
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    ax.plot(ccx_time, ccx, label="CalculiX full-order FEM", linewidth=2.1)
    ax.plot(time, py, "--", label="Python full-order FEM", linewidth=1.9)
    ax.set_xlabel("time")
    ax.set_ylabel("top-node Uz")
    ax.set_title(f"Full-order FEM external alignment, node {node_id + 1}")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_three_way_top_displacement(
    path: Path,
    node_id: int,
    dt: float,
    full,
    rom,
    calculix_history: CalculixNodeHistory,
) -> None:
    time = dt * (np.arange(full.displacement_history.shape[0]) + 1)
    py = full.displacement_history[:, 3 * node_id + 2]
    reduced = rom.displacement_history[:, 3 * node_id + 2]
    ccx = calculix_history.displacements[node_id][:, 2]
    ccx_time = calculix_history.times[: ccx.shape[0]]
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    ax.plot(ccx_time, ccx, label="CalculiX nonlinear contact full FEM", linewidth=2.1)
    ax.plot(time, py, "--", label="Python full FEM contact", linewidth=1.8)
    ax.plot(time, reduced, ":", label="Adaptive ROM contact", linewidth=2.2)
    ax.set_xlabel("time")
    ax.set_ylabel("top-node Uz")
    ax.set_title(f"Nonlinear contact three-way displacement, node {node_id + 1}")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_three_way_top_displacement_error(
    path: Path,
    node_id: int,
    dt: float,
    full,
    rom,
    calculix_history: CalculixNodeHistory,
) -> None:
    time = dt * (np.arange(full.displacement_history.shape[0]) + 1)
    py = full.displacement_history[:, 3 * node_id + 2]
    reduced = rom.displacement_history[:, 3 * node_id + 2]
    ccx = calculix_history.displacements[node_id][:, 2]
    ccx_time = calculix_history.times[: ccx.shape[0]]
    py_at_ccx = np.interp(ccx_time, time, py)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=False, constrained_layout=True)
    axes[0].plot(ccx_time, ccx, label="CalculiX nonlinear contact full FEM", linewidth=2.1)
    axes[0].plot(time, py, "--", label="Python full FEM contact", linewidth=1.8)
    axes[0].plot(time, reduced, ":", label="Adaptive ROM contact", linewidth=2.2)
    axes[0].set_ylabel("Uz")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(fontsize=8)

    axes[1].plot(time, reduced - py, color="tab:green", linewidth=1.8)
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_ylabel("Adaptive ROM - Python full")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(ccx_time, ccx - py_at_ccx, color="tab:blue", linewidth=1.8)
    axes[2].axhline(0.0, color="0.35", linewidth=0.8)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("CalculiX - Python full")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[2].grid(True, alpha=0.35)
    fig.suptitle(f"Nonlinear contact displacement residuals, node {node_id + 1}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def representative_node_displacement_error_summary(
    node_id: int,
    dt: float,
    full,
    rom,
    calculix_history: CalculixNodeHistory,
) -> dict[str, float | int]:
    time = dt * (np.arange(full.displacement_history.shape[0]) + 1)
    py = full.displacement_history[:, 3 * node_id + 2]
    reduced = rom.displacement_history[:, 3 * node_id + 2]
    ccx = calculix_history.displacements[node_id][:, 2]
    ccx_time = calculix_history.times[: ccx.shape[0]]
    py_at_ccx = np.interp(ccx_time, time, py)
    return {
        "node_id_zero_based": int(node_id),
        "node_id_one_based": int(node_id + 1),
        "adaptive_rom_minus_python_full_max_abs_uz": float(np.max(np.abs(reduced - py))),
        "adaptive_rom_minus_python_full_relative_l2_uz": relative_l2(py, reduced),
        "calculix_minus_python_full_max_abs_uz": float(np.max(np.abs(ccx - py_at_ccx))),
        "calculix_minus_python_full_relative_l2_uz": relative_l2(ccx, py_at_ccx),
    }


def compare_to_calculix_observed_displacements_at_times(
    result,
    dt: float,
    node_history: CalculixNodeHistory,
    observed_nodes: np.ndarray,
) -> CalculixObservedDisplacementSummary:
    python_times = dt * (np.arange(result.displacement_history.shape[0]) + 1)
    python_blocks: list[np.ndarray] = []
    calculix_blocks: list[np.ndarray] = []
    for node_id in np.asarray(observed_nodes, dtype=np.int64):
        if int(node_id) not in node_history.displacements:
            continue
        calculix_values = node_history.displacements[int(node_id)]
        count = min(calculix_values.shape[0], node_history.times.shape[0])
        if count == 0:
            continue
        times = node_history.times[:count]
        python_values = result.displacement_history[:, 3 * int(node_id) : 3 * int(node_id) + 3]
        interpolated = np.column_stack(
            [np.interp(times, python_times, python_values[:, component]) for component in range(3)]
        )
        python_blocks.append(interpolated)
        calculix_blocks.append(calculix_values[:count])
    if not python_blocks:
        raise ValueError("none of the requested observed nodes were present in CalculiX output")
    python_stack = np.vstack(python_blocks)
    calculix_stack = np.vstack(calculix_blocks)
    return CalculixObservedDisplacementSummary(
        relative_observed_displacement_l2=relative_l2(calculix_stack, python_stack),
        max_observed_displacement_error=float(np.max(np.abs(calculix_stack - python_stack))),
        sample_count=int(python_stack.shape[0]),
        node_count=len(python_blocks),
    )


def plot_contact_activity(
    path: Path,
    top_nodes: np.ndarray,
    nodes: np.ndarray,
    plane_z: float,
    dt: float,
    full,
    rom,
    calculix_history: CalculixNodeHistory,
) -> None:
    full_penetration = observed_penetration_history(full.displacement_history, top_nodes, nodes, plane_z)
    rom_penetration = observed_penetration_history(rom.displacement_history, top_nodes, nodes, plane_z)
    ccx_penetration = calculix_observed_penetration_history(calculix_history, top_nodes, nodes, plane_z)
    full_force = np.asarray([step.normal_force for step in full.steps], dtype=float)
    rom_force = np.asarray([step.normal_force for step in rom.steps], dtype=float)
    time = dt * (np.arange(full_penetration.shape[0]) + 1)
    ccx_time = calculix_history.times[: ccx_penetration.shape[0]]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(ccx_time, ccx_penetration, label="CalculiX observed penetration", linewidth=2.1)
    axes[0].plot(time, full_penetration, "--", label="Python full FEM penetration", linewidth=1.8)
    axes[0].plot(time, rom_penetration, ":", label="Adaptive ROM penetration", linewidth=2.2)
    axes[0].set_ylabel("max top penetration")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(fontsize=8)
    axes[1].plot(time, full_force, "--", label="Python full FEM normal force", linewidth=1.8)
    axes[1].plot(time, rom_force, ":", label="Adaptive ROM normal force", linewidth=2.2)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("normal contact force")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.suptitle("Contact activation evidence")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def observed_penetration_history(displacement_history: np.ndarray, top_nodes: np.ndarray, nodes: np.ndarray, plane_z: float) -> np.ndarray:
    deformed_z = nodes[top_nodes, 2][None, :] + displacement_history[:, 3 * top_nodes + 2]
    return np.maximum(np.max(deformed_z - plane_z, axis=1), 0.0)


def calculix_observed_penetration_history(
    calculix_history: CalculixNodeHistory,
    top_nodes: np.ndarray,
    nodes: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    histories = []
    for node_id in top_nodes:
        if int(node_id) in calculix_history.displacements:
            histories.append(nodes[int(node_id), 2] + calculix_history.displacements[int(node_id)][:, 2])
    if not histories:
        return np.zeros(0, dtype=float)
    count = min(history.shape[0] for history in histories)
    stacked = np.column_stack([history[:count] for history in histories])
    return np.maximum(np.max(stacked - plane_z, axis=1), 0.0)


def max_calculix_observed_penetration(
    calculix_history: CalculixNodeHistory,
    top_nodes: np.ndarray,
    nodes: np.ndarray,
    plane_z: float,
) -> float:
    history = calculix_observed_penetration_history(calculix_history, top_nodes, nodes, plane_z)
    return float(np.max(history)) if history.size else 0.0


def read_calculix_contact_blocks(path: Path) -> dict[str, float | int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cdis_header = re.compile(
        r"relative contact displacement .* time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)",
        re.IGNORECASE,
    )
    stress_header = re.compile(r"contact stress .* time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    block_count = 0
    row_count = 0
    max_pressure = 0.0
    max_overclosure = 0.0
    block_kind: str | None = None
    current_time: float | None = None
    cdis: dict[float, list[float]] = {}
    stress: dict[float, list[float]] = {}
    for line in lines:
        cdis_match = cdis_header.search(line)
        if cdis_match:
            block_kind = "cdis"
            current_time = float(cdis_match.group(1).replace("D", "E"))
            continue
        stress_match = stress_header.search(line)
        if stress_match:
            block_count += 1
            block_kind = "stress"
            current_time = float(stress_match.group(1).replace("D", "E"))
            continue
        if block_kind is None or current_time is None:
            continue
        if not line.strip():
            continue
        if line.lstrip().lower().startswith(("displacements", "relative contact", "contact stress")):
            block_kind = None
            continue
        parts = line.split()
        if len(parts) != 4:
            if len(parts) != 5:
                continue
        try:
            normal_column = 1 if len(parts) == 4 else 2
            normal_value = float(parts[normal_column].replace("D", "E"))
        except ValueError:
            continue
        if block_kind == "cdis":
            cdis.setdefault(current_time, []).append(normal_value)
            max_overclosure = max(max_overclosure, max(-normal_value, 0.0))
        elif block_kind == "stress":
            stress.setdefault(current_time, []).append(normal_value)
            row_count += 1
            max_pressure = max(max_pressure, abs(normal_value))

    overclosures = []
    pressures = []
    for time, pressure_values in stress.items():
        displacement_values = cdis.get(time, [])
        for displacement, pressure in zip(displacement_values, pressure_values, strict=False):
            overclosure = max(-float(displacement), 0.0)
            pressure = abs(float(pressure))
            if overclosure > 1.0e-14 and pressure > 0.0:
                overclosures.append(overclosure)
                pressures.append(pressure)
    if overclosures:
        overclosure_array = np.asarray(overclosures, dtype=float)
        pressure_array = np.asarray(pressures, dtype=float)
        equivalent = float((overclosure_array @ pressure_array) / (overclosure_array @ overclosure_array))
        max_ratio = float(max_pressure / max_overclosure) if max_overclosure > 0.0 else float("nan")
    else:
        equivalent = float("nan")
        max_ratio = float("nan")
    return {
        "block_count": block_count,
        "row_count": row_count,
        "max_pressure": max_pressure,
        "max_overclosure": max_overclosure,
        "sample_count": len(overclosures),
        "equivalent_pressure_stiffness": equivalent,
        "max_pressure_overclosure_ratio": max_ratio,
    }


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = float(np.linalg.norm(reference - candidate))
    denominator = max(float(np.linalg.norm(reference)), 1.0e-30)
    return numerator / denominator


if __name__ == "__main__":
    main()
