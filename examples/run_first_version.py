from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from modal_contact_rom.contact_modes import build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.reduced_dynamics import mass_orthonormalize
from modal_contact_rom.surface_patch import extract_surface, partition_surface_patches
from modal_contact_rom.validation import evaluate_compliance_errors, summarize_cases


def run() -> dict[str, object]:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=6)
    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(2, 2))
    loads = build_patch_normal_loads(fem.mesh, surface, patches)
    contact = solve_patch_contact_modes(fem.K, loads, fem.free_dofs)

    low_basis = mass_orthonormalize(fem.M, low.modes)
    combined_vectors = np.column_stack((low.modes, contact.modes))
    combined_basis = mass_orthonormalize(fem.M, combined_vectors)
    cases = evaluate_compliance_errors(
        fem.K,
        loads,
        {"low_modes": low_basis, "combined": combined_basis},
        fem.free_dofs,
    )
    summary = summarize_cases(cases)

    return {
        "nodes": fem.mesh.n_nodes,
        "elements": int(fem.mesh.elements.shape[0]),
        "surface_triangles": int(surface.triangles.shape[0]),
        "patches": len(patches),
        "low_mode_count": int(low_basis.shape[1]),
        "contact_mode_count": int(contact.modes.shape[1]),
        "combined_mode_count": int(combined_basis.shape[1]),
        "lowest_frequencies_hz": [float(x) for x in low.frequencies_hz[:6]],
        "summary": summary,
    }


def main() -> None:
    result = run()
    print(json.dumps(result, indent=2))
    low_error = result["summary"]["low_modes"]["mean_compliance_error"]  # type: ignore[index]
    combined_error = result["summary"]["combined"]["mean_compliance_error"]  # type: ignore[index]
    if not combined_error < 0.25 * low_error:
        raise SystemExit("combined patch basis did not improve the mean compliance error enough")


if __name__ == "__main__":
    main()

