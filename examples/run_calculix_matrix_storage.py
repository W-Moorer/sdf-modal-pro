from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.contact_modes import build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import (
    cantilever_block,
    matrix_storage_to_fem_system,
    read_matrix_storage,
    run_ccx_wsl,
    write_cantilever_matrix_storage_input,
)
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.reduced_dynamics import mass_orthonormalize
from modal_contact_rom.surface_patch import extract_surface, partition_surface_patches
from modal_contact_rom.validation import evaluate_compliance_errors, summarize_cases


def run() -> dict[str, object]:
    workdir = ROOT / "outputs" / "calculix_matrix_storage"
    job_name = "ccx_cantilever"
    base = cantilever_block(nx=3, ny=1, nz=1)
    input_path = write_cantilever_matrix_storage_input(
        workdir / f"{job_name}.inp",
        base.mesh,
        base.fixed_nodes,
        young_modulus=2.1e5,
        poisson_ratio=0.3,
        density=7.8e-9,
    )
    run_result = run_ccx_wsl(job_name, workdir)
    matrices = read_matrix_storage(job_name, workdir)
    fem = matrix_storage_to_fem_system(base.mesh, matrices)

    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=6)
    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(2, 2))
    loads = build_patch_normal_loads(fem.mesh, surface, patches)
    contact = solve_patch_contact_modes(fem.K, loads, fem.free_dofs)

    low_basis = mass_orthonormalize(fem.M, low.modes)
    combined_basis = mass_orthonormalize(fem.M, np.column_stack((low.modes, contact.modes)))
    cases = evaluate_compliance_errors(
        fem.K,
        loads,
        {"low_modes": low_basis, "combined": combined_basis},
        fem.free_dofs,
    )
    summary = summarize_cases(cases)

    return {
        "job": str(input_path),
        "ccx_returncode": run_result.returncode,
        "active_dofs": int(matrices.dof_map.full_dofs.size),
        "K_nnz": int(matrices.K.nnz),
        "M_nnz": int(matrices.M.nnz),
        "patches": len(patches),
        "contact_modes": int(contact.modes.shape[1]),
        "lowest_frequencies_hz": [float(x) for x in low.frequencies_hz[:6]],
        "summary": summary,
    }


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
