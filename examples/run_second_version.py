from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.contact_dynamics import AdaptiveModalContactSimulator, PrescribedRigidSphere, patch_mode_dict
from modal_contact_rom.contact_modes import build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.reduced_dynamics import compute_craig_bampton_basis, mass_orthonormalize
from modal_contact_rom.surface_patch import SurfacePatch, extract_surface, partition_surface_patches


def main() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2, stiffness_scale=100.0)
    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(2, 2))
    loads = build_patch_normal_loads(fem.mesh, surface, patches)
    contact = solve_patch_contact_modes(fem.K, loads, fem.free_dofs)
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
    mode_by_patch = patch_mode_dict(contact.patch_ids, contact.modes)

    target_patches = select_two_separated_contact_patches(patches)
    cases = []
    for patch in target_patches:
        center, radius = sphere_for_patch(patch, penetration=0.05)
        simulator = AdaptiveModalContactSimulator(
            fem=fem,
            surface=surface,
            patches=patches,
            base_basis=base_basis,
            patch_modes=mode_by_patch,
            rigid_sphere=PrescribedRigidSphere(radius=radius, center=center),
            penalty=250.0,
            contact_damping=1.5,
            rayleigh_alpha=0.05,
            rayleigh_beta=0.002,
            activation_count=2,
        )
        result = simulator.run(dt=0.01, steps=80)
        cases.append(summarize_case(patch, center, radius, result.active_patch_history, result.steps))

    print(
        json.dumps(
            {
                "craig_bampton": {
                    "interface_dofs": int(cb.interface_dofs.size),
                    "interior_dofs": int(cb.interior_dofs.size),
                    "constraint_modes": int(cb.constraint_modes.shape[1]),
                    "fixed_interface_modes": int(cb.fixed_interface_modes.shape[1]),
                    "base_basis_modes": int(base_basis.shape[1]),
                },
                "contact_mode_count": int(contact.modes.shape[1]),
                "cases": cases,
            },
            indent=2,
        )
    )


def select_two_separated_contact_patches(patches: list[SurfacePatch]) -> list[SurfacePatch]:
    contact_side = [patch for patch in patches if patch.key[0] == 0 and patch.key[1] == 1]
    if len(contact_side) < 2:
        raise RuntimeError("expected at least two patches on the +x contact side")
    first = min(contact_side, key=lambda patch: patch.centroid[1] + patch.centroid[2])
    second = max(contact_side, key=lambda patch: patch.centroid[1] + patch.centroid[2])
    return [first, second]


def sphere_for_patch(patch: SurfacePatch, penetration: float) -> tuple[np.ndarray, float]:
    radius = 1.5
    center = patch.centroid + patch.normal * (radius - penetration)
    return center, radius


def summarize_case(patch, center, radius, active_history, steps) -> dict[str, object]:
    active_counter = Counter(patch_id for active in active_history for patch_id in active)
    final = steps[-1]
    return {
        "target_patch_id": int(patch.patch_id),
        "sphere_center": [float(x) for x in center],
        "sphere_radius": float(radius),
        "most_common_active_patches": [[int(k), int(v)] for k, v in active_counter.most_common(4)],
        "target_was_activated": int(patch.patch_id) in active_counter,
        "min_gap": float(min(step.min_gap for step in steps)),
        "max_penetration": float(max(step.max_penetration for step in steps)),
        "max_normal_force": float(max(step.normal_force for step in steps)),
        "max_abs_energy_balance_error": float(max(abs(step.energy_balance_error) for step in steps)),
        "final_active_patch_ids": [int(x) for x in final.active_patch_ids],
        "final_elastic_energy": float(final.elastic_energy),
        "final_kinetic_energy": float(final.kinetic_energy),
        "final_contact_work": float(final.contact_work),
        "final_energy_balance_error": float(final.energy_balance_error),
    }


if __name__ == "__main__":
    main()
