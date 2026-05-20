from __future__ import annotations

import numpy as np

from modal_contact_rom.contact_modes import build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.reduced_dynamics import mass_orthonormalize
from modal_contact_rom.surface_patch import extract_surface, partition_surface_patches
from modal_contact_rom.validation import evaluate_compliance_errors, summarize_cases


def test_patch_contact_modes_improve_local_compliance() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=6)
    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(2, 2))
    loads = build_patch_normal_loads(fem.mesh, surface, patches)
    contact = solve_patch_contact_modes(fem.K, loads, fem.free_dofs)

    assert len(patches) > 0
    assert contact.modes.shape[1] > 0

    low_basis = mass_orthonormalize(fem.M, low.modes)
    combined_basis = mass_orthonormalize(fem.M, np.column_stack((low.modes, contact.modes)))
    cases = evaluate_compliance_errors(
        fem.K,
        loads,
        {"low_modes": low_basis, "combined": combined_basis},
        fem.free_dofs,
    )
    summary = summarize_cases(cases)

    assert summary["combined"]["mean_compliance_error"] < 0.25 * summary["low_modes"]["mean_compliance_error"]
    assert summary["combined"]["max_energy_error"] < 1.0e-6

