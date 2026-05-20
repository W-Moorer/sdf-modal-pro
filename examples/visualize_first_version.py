from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.contact_modes import PatchLoad, build_patch_normal_loads, solve_patch_contact_modes
from modal_contact_rom.fem_io import FEMSystem, Mesh, cantilever_block
from modal_contact_rom.modal_basis import compute_low_modes
from modal_contact_rom.reduced_dynamics import mass_orthonormalize, reduced_static_response
from modal_contact_rom.surface_patch import SurfaceMesh, extract_surface, partition_surface_patches
from modal_contact_rom.validation import ComplianceCase, evaluate_compliance_errors, summarize_cases

OUTPUT_DIR = ROOT / "outputs" / "first_version_validation"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_validation_case()

    paths = [
        plot_error_curve(
            data["cases"],
            "relative_compliance_error",
            "Relative Compliance Error by Contact Patch",
            "relative compliance error",
            OUTPUT_DIR / "compliance_error_by_patch.png",
        ),
        plot_error_curve(
            data["cases"],
            "relative_energy_error",
            "Relative Energy-Norm Error by Contact Patch",
            "relative energy-norm error",
            OUTPUT_DIR / "energy_error_by_patch.png",
        ),
    ]

    selected_patch_id = select_patch_with_largest_low_mode_error(data["cases"])
    selected = solve_selected_patch_case(data, selected_patch_id)
    paths.append(plot_displacement_clouds(data, selected, OUTPUT_DIR / f"surface_displacement_clouds_patch_{selected_patch_id:02d}.png"))
    paths.append(plot_strain_clouds(data, selected, OUTPUT_DIR / f"surface_equivalent_strain_clouds_patch_{selected_patch_id:02d}.png"))

    report = {
        "selected_patch_id": selected_patch_id,
        "summary": data["summary"],
        "selected_patch_errors": {
            case.basis_name: {
                "relative_compliance_error": case.relative_compliance_error,
                "relative_energy_error": case.relative_energy_error,
            }
            for case in data["cases"]
            if case.patch_id == selected_patch_id
        },
        "figures": [str(path) for path in paths],
        "note": (
            "The cloud plots are diagnostic displacement/equivalent-strain fields. "
            "This first-version K/M model is a spring-lattice surrogate, not a continuum "
            "constitutive FEM model, so physical stress clouds are intentionally not claimed."
        ),
    }
    report_path = OUTPUT_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


def build_validation_case() -> dict[str, object]:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    low = compute_low_modes(fem.K, fem.M, fem.free_dofs, num_modes=6)
    surface = extract_surface(fem.mesh)
    patches = partition_surface_patches(surface, bins=(2, 2))
    loads = build_patch_normal_loads(fem.mesh, surface, patches)
    contact = solve_patch_contact_modes(fem.K, loads, fem.free_dofs)

    low_basis = mass_orthonormalize(fem.M, low.modes)
    combined_basis = mass_orthonormalize(fem.M, np.column_stack((low.modes, contact.modes)))
    bases = {"low_modes": low_basis, "combined": combined_basis}
    cases = evaluate_compliance_errors(fem.K, loads, bases, fem.free_dofs)

    return {
        "fem": fem,
        "surface": surface,
        "patches": patches,
        "loads": loads,
        "bases": bases,
        "cases": cases,
        "summary": summarize_cases(cases),
    }


def select_patch_with_largest_low_mode_error(cases: list[ComplianceCase]) -> int:
    low_cases = [case for case in cases if case.basis_name == "low_modes"]
    if not low_cases:
        raise RuntimeError("no low_modes validation cases were produced")
    return max(low_cases, key=lambda case: case.relative_compliance_error).patch_id


def solve_selected_patch_case(data: dict[str, object], patch_id: int) -> dict[str, np.ndarray | PatchLoad]:
    fem = data["fem"]
    loads = data["loads"]
    bases = data["bases"]
    assert isinstance(fem, FEMSystem)
    assert isinstance(loads, list)
    assert isinstance(bases, dict)

    load = next(load for load in loads if load.patch_id == patch_id)
    Kff = fem.K[fem.free_dofs, :][:, fem.free_dofs].tocsc()
    exact = np.zeros(fem.mesh.n_dofs, dtype=float)
    exact[fem.free_dofs] = spla.factorized(Kff)(load.vector[fem.free_dofs])

    low = reduced_static_response(fem.K, bases["low_modes"], load.vector)
    combined = reduced_static_response(fem.K, bases["combined"], load.vector)
    return {
        "load": load,
        "exact": exact,
        "low_modes": low,
        "combined": combined,
        "low_residual": exact - low,
        "combined_residual": exact - combined,
    }


def plot_error_curve(
    cases: list[ComplianceCase],
    field_name: str,
    title: str,
    ylabel: str,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    basis_names = sorted({case.basis_name for case in cases})
    for basis_name in basis_names:
        basis_cases = sorted((case for case in cases if case.basis_name == basis_name), key=lambda case: case.patch_id)
        xs = [case.patch_id for case in basis_cases]
        ys = [max(float(getattr(case, field_name)), 1.0e-16) for case in basis_cases]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=basis_name)

    ax.set_yscale("log")
    ax.set_xlabel("patch id")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_displacement_clouds(data: dict[str, object], selected: dict[str, object], path: Path) -> Path:
    fem = data["fem"]
    surface = data["surface"]
    assert isinstance(fem, FEMSystem)
    assert isinstance(surface, SurfaceMesh)

    exact = selected["exact"]
    low = selected["low_modes"]
    combined = selected["combined"]
    low_residual = selected["low_residual"]
    combined_residual = selected["combined_residual"]
    assert isinstance(exact, np.ndarray)
    assert isinstance(low, np.ndarray)
    assert isinstance(combined, np.ndarray)
    assert isinstance(low_residual, np.ndarray)
    assert isinstance(combined_residual, np.ndarray)

    exact_mag = nodal_magnitude(exact)
    low_mag = nodal_magnitude(low)
    combined_mag = nodal_magnitude(combined)
    low_error_mag = nodal_magnitude(low_residual)
    combined_error_mag = nodal_magnitude(combined_residual)

    displacement_scale = deformation_scale(fem.mesh, exact)
    response_vmax = max(float(np.max(exact_mag)), float(np.max(low_mag)), float(np.max(combined_mag)), 1.0e-30)
    residual_vmax = max(float(np.max(low_error_mag)), float(np.max(combined_error_mag)), 1.0e-30)

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    plot_surface_scalar(
        axes[0],
        fem.mesh.nodes + displacement_scale * exact.reshape(-1, 3),
        surface.triangles,
        exact_mag,
        "full K solve |u|",
        "viridis",
        0.0,
        response_vmax,
    )
    plot_surface_scalar(
        axes[1],
        fem.mesh.nodes + displacement_scale * low.reshape(-1, 3),
        surface.triangles,
        low_mag,
        "low modes ROM |u|",
        "viridis",
        0.0,
        response_vmax,
    )
    plot_surface_scalar(
        axes[2],
        fem.mesh.nodes + displacement_scale * combined.reshape(-1, 3),
        surface.triangles,
        combined_mag,
        "low + patch modes ROM |u|",
        "viridis",
        0.0,
        response_vmax,
    )
    plot_surface_scalar(
        axes[3],
        fem.mesh.nodes,
        surface.triangles,
        low_error_mag,
        "low modes residual |u_exact-u_rom|",
        "magma",
        0.0,
        residual_vmax,
    )
    plot_surface_scalar(
        axes[4],
        fem.mesh.nodes,
        surface.triangles,
        combined_error_mag,
        "patch ROM residual |u_exact-u_rom|",
        "magma",
        0.0,
        residual_vmax,
    )
    axes[5].set_axis_off()
    fig.suptitle(f"Surface displacement diagnostic, deformation scale {displacement_scale:.3g}", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_strain_clouds(data: dict[str, object], selected: dict[str, object], path: Path) -> Path:
    fem = data["fem"]
    surface = data["surface"]
    assert isinstance(fem, FEMSystem)
    assert isinstance(surface, SurfaceMesh)

    exact = selected["exact"]
    low = selected["low_modes"]
    combined = selected["combined"]
    low_residual = selected["low_residual"]
    combined_residual = selected["combined_residual"]
    assert isinstance(exact, np.ndarray)
    assert isinstance(low, np.ndarray)
    assert isinstance(combined, np.ndarray)
    assert isinstance(low_residual, np.ndarray)
    assert isinstance(combined_residual, np.ndarray)

    exact_strain = nodal_equivalent_strain(fem.mesh, exact)
    low_strain = nodal_equivalent_strain(fem.mesh, low)
    combined_strain = nodal_equivalent_strain(fem.mesh, combined)
    low_residual_strain = nodal_equivalent_strain(fem.mesh, low_residual)
    combined_residual_strain = nodal_equivalent_strain(fem.mesh, combined_residual)

    displacement_scale = deformation_scale(fem.mesh, exact)
    strain_vmax = max(float(np.max(exact_strain)), float(np.max(low_strain)), float(np.max(combined_strain)), 1.0e-30)
    residual_vmax = max(float(np.max(low_residual_strain)), float(np.max(combined_residual_strain)), 1.0e-30)

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    plot_surface_scalar(
        axes[0],
        fem.mesh.nodes + displacement_scale * exact.reshape(-1, 3),
        surface.triangles,
        exact_strain,
        "full K solve eq. strain",
        "plasma",
        0.0,
        strain_vmax,
    )
    plot_surface_scalar(
        axes[1],
        fem.mesh.nodes + displacement_scale * low.reshape(-1, 3),
        surface.triangles,
        low_strain,
        "low modes ROM eq. strain",
        "plasma",
        0.0,
        strain_vmax,
    )
    plot_surface_scalar(
        axes[2],
        fem.mesh.nodes + displacement_scale * combined.reshape(-1, 3),
        surface.triangles,
        combined_strain,
        "low + patch modes ROM eq. strain",
        "plasma",
        0.0,
        strain_vmax,
    )
    plot_surface_scalar(
        axes[3],
        fem.mesh.nodes,
        surface.triangles,
        low_residual_strain,
        "low modes residual eq. strain",
        "inferno",
        0.0,
        residual_vmax,
    )
    plot_surface_scalar(
        axes[4],
        fem.mesh.nodes,
        surface.triangles,
        combined_residual_strain,
        "patch ROM residual eq. strain",
        "inferno",
        0.0,
        residual_vmax,
    )
    axes[5].set_axis_off()
    fig.suptitle("Surface equivalent-strain diagnostic from displacement gradient", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def nodal_magnitude(displacement: np.ndarray) -> np.ndarray:
    return np.linalg.norm(displacement.reshape(-1, 3), axis=1)


def deformation_scale(mesh: Mesh, displacement: np.ndarray) -> float:
    span = float(np.linalg.norm(np.max(mesh.nodes, axis=0) - np.min(mesh.nodes, axis=0)))
    max_disp = float(np.max(nodal_magnitude(displacement)))
    if max_disp <= 0.0:
        return 1.0
    return 0.18 * span / max_disp


def nodal_equivalent_strain(mesh: Mesh, displacement: np.ndarray) -> np.ndarray:
    u_nodes = displacement.reshape(-1, 3)
    accum = np.zeros(mesh.n_nodes, dtype=float)
    weights = np.zeros(mesh.n_nodes, dtype=float)

    for element in mesh.elements:
        x = mesh.nodes[element]
        u = u_nodes[element]
        a = np.column_stack((np.ones(element.shape[0]), x))
        coeffs, *_ = np.linalg.lstsq(a, u, rcond=None)
        grad = coeffs[1:, :].T
        strain = 0.5 * (grad + grad.T)
        trace = float(np.trace(strain))
        deviator = strain - np.eye(3) * (trace / 3.0)
        eq_strain = float(np.sqrt(max((2.0 / 3.0) * np.sum(deviator * deviator), 0.0)))
        accum[element] += eq_strain
        weights[element] += 1.0

    valid = weights > 0.0
    result = np.zeros(mesh.n_nodes, dtype=float)
    result[valid] = accum[valid] / weights[valid]
    return result


def plot_surface_scalar(
    ax: plt.Axes,
    nodes: np.ndarray,
    triangles: np.ndarray,
    nodal_scalar: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    verts = nodes[triangles]
    face_values = np.mean(nodal_scalar[triangles], axis=1)
    collection = Poly3DCollection(verts, linewidth=0.15, edgecolor=(0.08, 0.08, 0.08, 0.22))
    collection.set_array(face_values)
    collection.set_cmap(cmap)
    collection.set_clim(vmin, vmax)
    ax.add_collection3d(collection)
    set_equal_3d_limits(ax, nodes)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.tick_params(labelsize=7)
    plt.colorbar(collection, ax=ax, shrink=0.6, pad=0.04)


def set_equal_3d_limits(ax: plt.Axes, nodes: np.ndarray) -> None:
    mins = np.min(nodes, axis=0)
    maxs = np.max(nodes, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


if __name__ == "__main__":
    main()
