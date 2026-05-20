from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

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


def _has_wsl_ccx() -> bool:
    if shutil.which("wsl") is None:
        return False
    completed = subprocess.run(
        ["wsl", "bash", "-lc", "command -v ccx >/dev/null"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return completed.returncode == 0


@pytest.mark.skipif(not _has_wsl_ccx(), reason="WSL CalculiX ccx is not available")
def test_wsl_calculix_matrix_storage_roundtrip(tmp_path) -> None:
    fem_seed = cantilever_block(nx=1, ny=1, nz=1)
    job_name = "matrix_roundtrip"
    write_cantilever_matrix_storage_input(tmp_path / f"{job_name}.inp", fem_seed.mesh, fem_seed.fixed_nodes)

    result = run_ccx_wsl(job_name, tmp_path)
    matrices = read_matrix_storage(job_name, tmp_path)
    fem = matrix_storage_to_fem_system(fem_seed.mesh, matrices)

    assert result.returncode == 0
    assert matrices.K.shape == matrices.M.shape
    assert matrices.K.nnz > 0
    assert matrices.M.nnz > 0
    assert fem.free_dofs.size == matrices.K.shape[0]
    assert np.all(np.diff(matrices.dof_map.full_dofs) >= 0)
    assert np.linalg.norm((matrices.K - matrices.K.T).toarray()) < 1.0e-6 * max(np.linalg.norm(matrices.K.toarray()), 1.0)

    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=4)
    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(1, 1))
    loads = build_patch_normal_loads(fem.mesh, surface, patches)
    contact = solve_patch_contact_modes(fem.K, loads, fem.free_dofs)
    low_basis = mass_orthonormalize(fem.M, low.modes)
    combined_basis = mass_orthonormalize(fem.M, np.column_stack((low.modes, contact.modes)))
    summary = summarize_cases(
        evaluate_compliance_errors(
            fem.K,
            loads,
            {"low_modes": low_basis, "combined": combined_basis},
            fem.free_dofs,
        )
    )
    assert summary["combined"]["mean_compliance_error"] < 1.0e-10
    assert summary["combined"]["mean_compliance_error"] < summary["low_modes"]["mean_compliance_error"]
