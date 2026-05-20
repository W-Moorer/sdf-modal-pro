from __future__ import annotations

from dataclasses import asdict
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.contact_dynamics import run_full_forced_response, run_reduced_forced_response
from modal_contact_rom.contact_modes import build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import (
    cantilever_block,
    matrix_storage_to_fem_system,
    read_matrix_storage,
    read_node_print_displacements,
    run_ccx_wsl,
    write_cantilever_dynamic_cload_input,
    write_cantilever_matrix_storage_input,
)
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.reduced_dynamics import compute_craig_bampton_basis, mass_orthonormalize
from modal_contact_rom.surface_patch import extract_surface, partition_surface_patches
from modal_contact_rom.validation import compare_forced_results, compare_to_calculix_observed_displacements


def main() -> None:
    result = run_three_way_benchmark()
    print(json.dumps(result, indent=2))


def run_three_way_benchmark() -> dict[str, object]:
    workdir = ROOT / "outputs" / "calculix_three_way"
    workdir.mkdir(parents=True, exist_ok=True)
    matrix_job = "three_way_mtx"
    dynamic_job = "three_way_dynamic"
    young_modulus = 1000.0
    poisson_ratio = 0.3
    density = 1.0
    dt = 0.005
    steps = 80

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
    fem = matrix_storage_to_fem_system(seed.mesh, matrices)

    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(1, 1))
    target_patch = next(patch for patch in patches if patch.key[0] == 2 and patch.key[1] == 1)
    all_loads = build_patch_normal_loads(fem.mesh, surface, patches)
    target_load = next(load for load in all_loads if load.patch_id == target_patch.patch_id)
    force = target_load.vector
    observed_nodes = target_patch.node_indices

    write_cantilever_dynamic_cload_input(
        workdir / f"{dynamic_job}.inp",
        seed.mesh,
        seed.fixed_nodes,
        force,
        observed_nodes,
        time_step=dt,
        total_time=dt * steps,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        max_increments=steps + 5,
    )
    run_ccx_wsl(dynamic_job, workdir)
    calculix_history = read_node_print_displacements(workdir / f"{dynamic_job}.dat")

    python_full = run_full_forced_response(fem, force, dt, steps)
    contact_modes = solve_patch_contact_modes(fem.K, all_loads, fem.free_dofs)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=6)
    interface_nodes = np.flatnonzero(np.isclose(fem.mesh.nodes[:, 0], fem.mesh.nodes[:, 0].max()))
    cb = compute_craig_bampton_basis(
        fem.K,
        fem.M,
        fem.free_dofs,
        fem.mesh.dof_indices(interface_nodes),
        num_fixed_interface_modes=4,
    )
    target_mode_index = int(np.flatnonzero(contact_modes.patch_ids == target_patch.patch_id)[0])
    adaptive_basis = mass_orthonormalize(
        fem.M,
        np.column_stack((cb.basis, low.modes, contact_modes.modes[:, target_mode_index])),
    )
    adaptive_rom = run_reduced_forced_response(fem, adaptive_basis, force, dt, steps)

    full_vs_calculix = compare_to_calculix_observed_displacements(python_full, calculix_history, observed_nodes)
    rom_vs_full = compare_forced_results(python_full, adaptive_rom)
    rom_vs_calculix = compare_to_calculix_observed_displacements(adaptive_rom, calculix_history, observed_nodes)

    figure_path = workdir / "three_way_observed_displacement.png"
    plot_three_way_observed_displacement(
        figure_path,
        observed_nodes[-1],
        dt,
        python_full,
        adaptive_rom,
        calculix_history,
    )

    report = {
        "external_solver": "CalculiX ccx via WSL",
        "matrix_storage_input": str(workdir / f"{matrix_job}.inp"),
        "direct_dynamic_input": str(workdir / f"{dynamic_job}.inp"),
        "target_patch_id": int(target_patch.patch_id),
        "target_patch_key": list(target_patch.key),
        "observed_nodes": [int(node_id) for node_id in observed_nodes],
        "active_rom_basis_modes": int(adaptive_basis.shape[1]),
        "full_free_dofs": int(fem.free_dofs.size),
        "python_full_vs_calculix": asdict(full_vs_calculix),
        "adaptive_rom_vs_python_full": asdict(rom_vs_full),
        "adaptive_rom_vs_calculix": asdict(rom_vs_calculix),
        "figure": str(figure_path),
    }
    (workdir / "three_way_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def plot_three_way_observed_displacement(
    path: Path,
    node_id: int,
    dt: float,
    python_full,
    adaptive_rom,
    calculix_history,
) -> None:
    py = python_full.displacement_history[:, 3 * node_id : 3 * node_id + 3]
    rom = adaptive_rom.displacement_history[:, 3 * node_id : 3 * node_id + 3]
    ccx = calculix_history.displacements[node_id]
    count = min(py.shape[0], rom.shape[0], ccx.shape[0])
    time = dt * (np.arange(count) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    labels = ("Ux", "Uy", "Uz")
    for component, ax in enumerate(axes):
        ax.plot(time, ccx[:count, component], label="CalculiX full FEM", linewidth=2.0)
        ax.plot(time, py[:count, component], "--", label="Python full FEM", linewidth=1.7)
        ax.plot(time, rom[:count, component], ":", label="Adaptive ROM", linewidth=2.0)
        ax.set_ylabel(labels[component])
        ax.grid(True, alpha=0.35)
    axes[-1].set_xlabel("time")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Three-way dynamic displacement comparison, node {node_id + 1}", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
