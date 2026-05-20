from __future__ import annotations

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

from examples.run_second_version import select_two_separated_contact_patches, sphere_for_patch
from modal_contact_rom.contact_dynamics import AdaptiveModalContactSimulator, PrescribedRigidSphere, patch_mode_dict
from modal_contact_rom.contact_modes import build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.reduced_dynamics import compute_craig_bampton_basis, mass_orthonormalize
from modal_contact_rom.surface_patch import extract_surface, partition_surface_patches


def main() -> None:
    output_dir = ROOT / "outputs" / "second_version_dynamics"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    figures = []
    reports = []
    for patch in select_two_separated_contact_patches(patches):
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
        figure_path = output_dir / f"dynamics_patch_{patch.patch_id:02d}.png"
        plot_history(patch.patch_id, result.steps, figure_path)
        figures.append(str(figure_path))
        reports.append(
            {
                "target_patch_id": int(patch.patch_id),
                "final_active_patch_ids": [int(x) for x in result.steps[-1].active_patch_ids],
                "max_force": float(max(step.normal_force for step in result.steps)),
                "min_gap": float(min(step.min_gap for step in result.steps)),
                "max_abs_energy_balance_error": float(max(abs(step.energy_balance_error) for step in result.steps)),
            }
        )

    report = {"figures": figures, "cases": reports}
    (output_dir / "second_version_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def plot_history(target_patch_id: int, steps, path: Path) -> None:
    time = np.asarray([step.time for step in steps])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(time, [step.min_gap for step in steps], color="tab:blue")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("gap")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("min gap")
    axes[0, 0].grid(True, alpha=0.35)

    axes[0, 1].plot(time, [step.normal_force for step in steps], color="tab:red")
    axes[0, 1].set_title("contact force")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("normal force")
    axes[0, 1].grid(True, alpha=0.35)

    axes[1, 0].plot(time, [step.elastic_energy for step in steps], label="elastic")
    axes[1, 0].plot(time, [step.kinetic_energy for step in steps], label="kinetic")
    axes[1, 0].plot(time, [step.contact_work for step in steps], label="contact work")
    axes[1, 0].plot(time, [step.energy_balance_error for step in steps], label="balance error")
    axes[1, 0].set_title("energy diagnostic")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.35)

    xs = []
    ys = []
    for step in steps:
        for patch_id in step.active_patch_ids:
            xs.append(step.time)
            ys.append(patch_id)
    axes[1, 1].scatter(xs, ys, s=14, color="tab:green")
    axes[1, 1].axhline(target_patch_id, color="black", linewidth=0.8, linestyle="--", label="target")
    axes[1, 1].set_title("online active patches")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("patch id")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.35)

    fig.suptitle(f"Second-version contact dynamics, target patch {target_patch_id}", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
