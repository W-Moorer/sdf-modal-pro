from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from modal_contact_rom.contact_modes.loads import PatchLoad
from modal_contact_rom.reduced_dynamics.basis import reduced_static_response


@dataclass(frozen=True)
class ComplianceCase:
    basis_name: str
    patch_id: int
    exact_compliance: float
    reduced_compliance: float
    relative_compliance_error: float
    relative_energy_error: float


def evaluate_compliance_errors(
    K: sp.spmatrix,
    loads: list[PatchLoad],
    bases: dict[str, np.ndarray],
    free_dofs: np.ndarray,
    min_rhs_norm: float = 1.0e-12,
) -> list[ComplianceCase]:
    """Compare exact constrained static response against reduced bases."""

    free_dofs = np.asarray(free_dofs, dtype=np.int64)
    Kff = K[free_dofs, :][:, free_dofs].tocsc()
    solve = spla.factorized(Kff)
    cases: list[ComplianceCase] = []

    for load in loads:
        rhs = load.vector[free_dofs]
        if np.linalg.norm(rhs) <= min_rhs_norm:
            continue
        exact = np.zeros(K.shape[0], dtype=float)
        exact[free_dofs] = solve(rhs)
        exact_compliance = float(load.vector @ exact)
        exact_energy = max(float(exact @ (K @ exact)), 1.0e-30)

        for basis_name, basis in bases.items():
            reduced = reduced_static_response(K, basis, load.vector)
            reduced_compliance = float(load.vector @ reduced)
            residual = exact - reduced
            energy_error = float(np.sqrt(max(residual @ (K @ residual), 0.0) / exact_energy))
            compliance_error = abs(exact_compliance - reduced_compliance) / max(abs(exact_compliance), 1.0e-30)
            cases.append(
                ComplianceCase(
                    basis_name=basis_name,
                    patch_id=load.patch_id,
                    exact_compliance=exact_compliance,
                    reduced_compliance=reduced_compliance,
                    relative_compliance_error=float(compliance_error),
                    relative_energy_error=energy_error,
                )
            )
    return cases


def summarize_cases(cases: list[ComplianceCase]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[ComplianceCase]] = defaultdict(list)
    for case in cases:
        grouped[case.basis_name].append(case)

    summary: dict[str, dict[str, float]] = {}
    for basis_name, items in grouped.items():
        comp = np.asarray([item.relative_compliance_error for item in items], dtype=float)
        energy = np.asarray([item.relative_energy_error for item in items], dtype=float)
        summary[basis_name] = {
            "case_count": float(len(items)),
            "mean_compliance_error": float(np.mean(comp)),
            "max_compliance_error": float(np.max(comp)),
            "mean_energy_error": float(np.mean(energy)),
            "max_energy_error": float(np.max(energy)),
        }
    return summary

