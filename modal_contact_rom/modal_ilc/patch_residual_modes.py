from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from modal_contact_rom.modal_ilc.patch_load_basis import PatchLoadBasis, assemble_load_basis


@dataclass(frozen=True)
class PatchResidualBlock:
    """Residual deformation block associated with one active contact patch."""

    patch_id: int
    Phi_B_j: np.ndarray

    def __post_init__(self) -> None:
        modes = np.asarray(self.Phi_B_j, dtype=float)
        if modes.ndim != 2:
            raise ValueError("Phi_B_j must be a two-dimensional residual mode matrix")
        if modes.shape[0] == 0:
            raise ValueError("Phi_B_j must have at least one displacement DOF")
        object.__setattr__(self, "patch_id", int(self.patch_id))
        object.__setattr__(self, "Phi_B_j", modes)

    @property
    def n_dofs(self) -> int:
        return int(self.Phi_B_j.shape[0])

    @property
    def alpha_dimension(self) -> int:
        return int(self.Phi_B_j.shape[1])

    @property
    def modes(self) -> np.ndarray:
        return self.Phi_B_j

    def deformation(self, alpha: np.ndarray) -> np.ndarray:
        alpha_vector = np.asarray(alpha, dtype=float)
        if alpha_vector.ndim != 1:
            raise ValueError("alpha must be a one-dimensional vector")
        if alpha_vector.size != self.alpha_dimension:
            raise ValueError("alpha length must match the residual block dimension")
        return self.Phi_B_j @ alpha_vector


@dataclass(frozen=True)
class PatchResidualBasis:
    """Offline residual-mode basis for patch ILC static correction."""

    load_bases: tuple[PatchLoadBasis, ...]
    blocks: tuple[PatchResidualBlock, ...]
    static_responses: np.ndarray
    modal_static_responses: np.ndarray
    residual_modes: np.ndarray
    retained_modes: np.ndarray
    K_kk: np.ndarray

    def __post_init__(self) -> None:
        static = np.asarray(self.static_responses, dtype=float)
        modal = np.asarray(self.modal_static_responses, dtype=float)
        residual = np.asarray(self.residual_modes, dtype=float)
        retained = np.asarray(self.retained_modes, dtype=float)
        K_kk = np.asarray(self.K_kk, dtype=float)
        if static.ndim != 2:
            raise ValueError("static_responses must be a two-dimensional matrix")
        if modal.shape != static.shape or residual.shape != static.shape:
            raise ValueError("modal and residual response matrices must match static_responses")
        if retained.ndim != 2 or retained.shape[0] != static.shape[0]:
            raise ValueError("retained_modes must be a two-dimensional full-DOF matrix")
        if K_kk.shape != (retained.shape[1], retained.shape[1]):
            raise ValueError("K_kk shape must match retained mode count")
        if sum(block.alpha_dimension for block in self.blocks) != static.shape[1]:
            raise ValueError("block dimensions must cover all residual columns")
        object.__setattr__(self, "static_responses", static)
        object.__setattr__(self, "modal_static_responses", modal)
        object.__setattr__(self, "residual_modes", residual)
        object.__setattr__(self, "retained_modes", retained)
        object.__setattr__(self, "K_kk", K_kk)

    @property
    def alpha_dimension(self) -> int:
        return int(self.static_responses.shape[1])

    @property
    def n_dofs(self) -> int:
        return int(self.static_responses.shape[0])

    @property
    def load_matrix(self) -> np.ndarray:
        return assemble_load_basis(self.load_bases)

    def static_response(self, alpha: np.ndarray) -> np.ndarray:
        return self.static_responses @ _as_alpha(alpha, self.alpha_dimension)

    def modal_static_response(self, alpha: np.ndarray) -> np.ndarray:
        return self.modal_static_responses @ _as_alpha(alpha, self.alpha_dimension)

    def residual_response(self, alpha: np.ndarray) -> np.ndarray:
        return self.residual_modes @ _as_alpha(alpha, self.alpha_dimension)

    def reconstructed_response(self, alpha: np.ndarray) -> np.ndarray:
        alpha_vector = _as_alpha(alpha, self.alpha_dimension)
        return self.modal_static_response(alpha_vector) + self.residual_response(alpha_vector)


def build_patch_residual_basis(
    K: sp.spmatrix,
    load_bases: list[PatchLoadBasis] | tuple[PatchLoadBasis, ...],
    retained_modes: np.ndarray,
    free_dofs: np.ndarray | None = None,
) -> PatchResidualBasis:
    """Build ``Phi_B = G_B - Psi K_kk^-1 Psi.T B`` for patch load bases."""

    load_basis_tuple = tuple(load_bases)
    B = assemble_load_basis(load_basis_tuple)
    if B.shape[0] != K.shape[0]:
        raise ValueError("load basis row count must match K")

    Psi = np.asarray(retained_modes, dtype=float)
    if Psi.ndim != 2 or Psi.shape[0] != K.shape[0]:
        raise ValueError("retained_modes must have shape (K.shape[0], n_modes)")

    G_B = solve_static_load_responses(K, B, free_dofs=free_dofs)
    modal_static, K_kk = modal_static_contribution(K, Psi, B)
    Phi_B = G_B - modal_static

    blocks: list[PatchResidualBlock] = []
    cursor = 0
    for basis in load_basis_tuple:
        next_cursor = cursor + basis.load_dimension
        blocks.append(PatchResidualBlock(basis.patch_id, Phi_B[:, cursor:next_cursor]))
        cursor = next_cursor

    return PatchResidualBasis(
        load_bases=load_basis_tuple,
        blocks=tuple(blocks),
        static_responses=G_B,
        modal_static_responses=modal_static,
        residual_modes=Phi_B,
        retained_modes=Psi,
        K_kk=K_kk,
    )


def solve_static_load_responses(
    K: sp.spmatrix,
    loads: np.ndarray,
    free_dofs: np.ndarray | None = None,
) -> np.ndarray:
    """Solve constrained static attachment responses ``K G = B``."""

    B = np.asarray(loads, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    if B.ndim != 2:
        raise ValueError("loads must be a one- or two-dimensional array")
    if K.shape[0] != K.shape[1] or K.shape[0] != B.shape[0]:
        raise ValueError("K must be square and match the load vector length")

    if free_dofs is None:
        solution = spla.spsolve(K.tocsc(), B)
        solution = np.asarray(solution, dtype=float)
        return solution.reshape(K.shape[0], B.shape[1])

    free = np.asarray(free_dofs, dtype=np.int64).reshape(-1)
    Kff = K[free, :][:, free].tocsc()
    Bf = B[free, :]
    solver = spla.factorized(Kff)
    Gf = np.column_stack([solver(Bf[:, column]) for column in range(Bf.shape[1])])
    G = np.zeros_like(B)
    G[free, :] = Gf
    return G


def modal_static_contribution(
    K: sp.spmatrix,
    retained_modes: np.ndarray,
    loads: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``Psi K_kk^-1 Psi.T B`` and ``K_kk``."""

    Psi = np.asarray(retained_modes, dtype=float)
    B = np.asarray(loads, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    if Psi.ndim != 2 or Psi.shape[0] != K.shape[0]:
        raise ValueError("retained_modes must have shape (K.shape[0], n_modes)")
    if B.ndim != 2 or B.shape[0] != K.shape[0]:
        raise ValueError("loads must have shape (K.shape[0], n_loads)")
    if Psi.shape[1] == 0:
        return np.zeros_like(B), np.zeros((0, 0), dtype=float)

    K_kk = Psi.T @ (K @ Psi)
    K_kk = 0.5 * (K_kk + K_kk.T)
    rhs = Psi.T @ B
    try:
        coeffs = la.solve(K_kk, rhs, assume_a="sym")
    except la.LinAlgError:
        coeffs = la.pinvh(K_kk) @ rhs
    return Psi @ coeffs, K_kk


def static_completeness_errors(residual_basis: PatchResidualBasis) -> np.ndarray:
    """Per-column relative error of modal-plus-residual reconstruction."""

    reconstructed = residual_basis.modal_static_responses + residual_basis.residual_modes
    return _relative_column_errors(residual_basis.static_responses, reconstructed)


def compliance_errors(residual_basis: PatchResidualBasis) -> tuple[np.ndarray, np.ndarray]:
    """Return low-mode-only and residual-corrected relative compliance errors."""

    B = residual_basis.load_matrix
    static_compliance = np.sum(B * residual_basis.static_responses, axis=0)
    low_compliance = np.sum(B * residual_basis.modal_static_responses, axis=0)
    reconstructed = residual_basis.modal_static_responses + residual_basis.residual_modes
    residual_compliance = np.sum(B * reconstructed, axis=0)
    denom = np.maximum(np.abs(static_compliance), 1.0e-30)
    low_error = np.abs(low_compliance - static_compliance) / denom
    residual_error = np.abs(residual_compliance - static_compliance) / denom
    return low_error, residual_error


def write_patch_residual_tables(residual_basis: PatchResidualBasis, output_dir: str | Path) -> None:
    """Write Phase-4 residual static-completeness diagnostics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    completeness = static_completeness_errors(residual_basis)
    low_error, residual_error = compliance_errors(residual_basis)
    load_bases = residual_basis.load_bases

    static_lines = ["patch_id,basis_column,relative_static_error\n"]
    compliance_lines = [
        "patch_id,basis_column,static_compliance,low_relative_error,residual_relative_error\n"
    ]
    B = residual_basis.load_matrix
    static_compliance = np.sum(B * residual_basis.static_responses, axis=0)
    cursor = 0
    for basis in load_bases:
        for local_column in range(basis.load_dimension):
            static_lines.append(f"{basis.patch_id},{local_column},{completeness[cursor]:.17g}\n")
            compliance_lines.append(
                f"{basis.patch_id},{local_column},{static_compliance[cursor]:.17g},"
                f"{low_error[cursor]:.17g},{residual_error[cursor]:.17g}\n"
            )
            cursor += 1

    compression_lines = [
        "stored_residual_columns,total_residual_columns,compression_ratio,max_relative_compliance_error\n",
        f"{residual_basis.alpha_dimension},{residual_basis.alpha_dimension},1,0\n",
    ]
    summary = (
        "# Patch Residual Summary\n\n"
        f"Residual columns: {residual_basis.alpha_dimension}\n"
        f"Max static completeness error: {float(np.max(completeness)):.6g}\n"
        f"Mean low-mode compliance error: {float(np.mean(low_error)):.6g}\n"
        f"Max residual compliance error: {float(np.max(residual_error)):.6g}\n"
    )
    (output_path / "residual_static_error.csv").write_text("".join(static_lines), encoding="utf-8")
    (output_path / "patch_compliance_error.csv").write_text("".join(compliance_lines), encoding="utf-8")
    (output_path / "compression_error.csv").write_text("".join(compression_lines), encoding="utf-8")
    (output_path / "residual_summary.md").write_text(summary, encoding="utf-8")


def _relative_column_errors(reference: np.ndarray, approximation: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(reference, axis=0)
    errors = np.linalg.norm(approximation - reference, axis=0)
    return errors / np.maximum(norms, 1.0e-30)


def _as_alpha(alpha: np.ndarray, expected_size: int) -> np.ndarray:
    alpha_vector = np.asarray(alpha, dtype=float)
    if alpha_vector.ndim != 1:
        raise ValueError("alpha must be a one-dimensional vector")
    if alpha_vector.size != expected_size:
        raise ValueError("alpha length must match residual basis dimension")
    return alpha_vector
