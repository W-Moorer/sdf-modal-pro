from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.surface_patch.extract import SurfaceMesh, SurfacePatch


@dataclass(frozen=True)
class PatchLoadBasis:
    """Patch-level interface load coordinate basis.

    ``B_j`` maps the patch ILC amplitudes for one patch into the full force
    vector. It is part of the contact/projection layer, not the dynamic state.
    """

    patch_id: int
    basis_type: str
    B_j: np.ndarray
    labels: tuple[str, ...] | None = None
    node_indices: np.ndarray | None = None
    node_weights: np.ndarray | None = None
    patch_area: float | None = None
    resultant: np.ndarray | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.B_j, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("B_j must be a two-dimensional load basis matrix")
        if matrix.shape[0] == 0:
            raise ValueError("B_j must have at least one force DOF")
        if not self.basis_type:
            raise ValueError("basis_type must be non-empty")
        if self.labels is not None and len(self.labels) != matrix.shape[1]:
            raise ValueError("labels length must match the number of load basis columns")
        if self.node_indices is not None:
            node_indices = np.asarray(self.node_indices, dtype=np.int64)
            if node_indices.ndim != 1:
                raise ValueError("node_indices must be a one-dimensional vector")
            object.__setattr__(self, "node_indices", node_indices)
        if self.node_weights is not None:
            node_weights = np.asarray(self.node_weights, dtype=float)
            if node_weights.ndim != 1:
                raise ValueError("node_weights must be a one-dimensional vector")
            if self.node_indices is not None and node_weights.size != self.node_indices.size:
                raise ValueError("node_weights length must match node_indices length")
            if np.any(node_weights < -1.0e-14):
                raise ValueError("node_weights must be non-negative")
            object.__setattr__(self, "node_weights", node_weights)
        if self.resultant is not None:
            resultant = np.asarray(self.resultant, dtype=float)
            if resultant.shape != (3,):
                raise ValueError("resultant must have shape (3,)")
            object.__setattr__(self, "resultant", resultant)
        object.__setattr__(self, "patch_id", int(self.patch_id))
        object.__setattr__(self, "B_j", matrix)

    @property
    def n_dofs(self) -> int:
        return int(self.B_j.shape[0])

    @property
    def load_dimension(self) -> int:
        return int(self.B_j.shape[1])

    @property
    def matrix(self) -> np.ndarray:
        return self.B_j


def assemble_load_basis(load_bases: list[PatchLoadBasis] | tuple[PatchLoadBasis, ...]) -> np.ndarray:
    """Column-stack active patch load bases into ``B_A``."""

    if not load_bases:
        return np.zeros((0, 0), dtype=float)

    n_dofs = load_bases[0].n_dofs
    for basis in load_bases:
        if basis.n_dofs != n_dofs:
            raise ValueError("all patch load bases must have the same number of force DOFs")

    return np.column_stack([basis.B_j for basis in load_bases])


def build_normal_patch_load_basis(mesh: Mesh, surface: SurfaceMesh, patch: SurfacePatch) -> PatchLoadBasis:
    """Build the Phase-3 unit-resultant normal load basis for one patch."""

    node_area = _patch_nodal_area_weights(surface, patch)
    total_area = float(np.sum(list(node_area.values())))
    if total_area <= 0.0:
        raise ValueError(f"patch {patch.patch_id} has zero surface area")

    normal = np.asarray(patch.normal, dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        raise ValueError(f"patch {patch.patch_id} has zero normal")
    normal = normal / normal_norm

    node_indices = np.asarray(sorted(node_area), dtype=np.int64)
    node_weights = np.asarray([node_area[int(node_id)] / total_area for node_id in node_indices], dtype=float)
    B_j = np.zeros((mesh.n_dofs, 1), dtype=float)
    for node_id, weight in zip(node_indices, node_weights):
        start = 3 * int(node_id)
        B_j[start : start + 3, 0] = weight * normal

    resultant = np.sum(B_j.reshape(mesh.n_nodes, 3, 1), axis=0)[:, 0]
    return PatchLoadBasis(
        patch_id=patch.patch_id,
        basis_type="normal",
        B_j=B_j,
        labels=("normal",),
        node_indices=node_indices,
        node_weights=node_weights,
        patch_area=patch.area,
        resultant=resultant,
    )


def build_normal_patch_load_bases(
    mesh: Mesh,
    surface: SurfaceMesh,
    patches: list[SurfacePatch] | tuple[SurfacePatch, ...],
) -> list[PatchLoadBasis]:
    """Build one unit normal load basis column for each patch."""

    return [build_normal_patch_load_basis(mesh, surface, patch) for patch in patches]


def load_basis_resultants(load_bases: list[PatchLoadBasis] | tuple[PatchLoadBasis, ...]) -> np.ndarray:
    """Return one 3D resultant vector per load basis column."""

    if not load_bases:
        return np.zeros((0, 3), dtype=float)
    resultants = []
    for basis in load_bases:
        if basis.n_dofs % 3 != 0:
            raise ValueError("load basis DOFs must be divisible by three")
        vectors = basis.B_j.reshape(basis.n_dofs // 3, 3, basis.load_dimension)
        for column in range(basis.load_dimension):
            resultants.append(np.sum(vectors[:, :, column], axis=0))
    return np.asarray(resultants, dtype=float)


def load_basis_condition_number(
    load_bases: list[PatchLoadBasis] | tuple[PatchLoadBasis, ...],
    weights: np.ndarray | None = None,
) -> float:
    """Condition number of ``B_A.T W B_A`` for a patch load basis set."""

    gram = load_basis_gram(load_bases, weights)
    if gram.size == 0:
        return 1.0
    return float(np.linalg.cond(gram))


def load_basis_gram(
    load_bases: list[PatchLoadBasis] | tuple[PatchLoadBasis, ...],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    B = assemble_load_basis(load_bases)
    if B.size == 0:
        return np.zeros((0, 0), dtype=float)
    if weights is None:
        return B.T @ B
    W = np.asarray(weights, dtype=float)
    if W.ndim == 1:
        if W.size != B.shape[0]:
            raise ValueError("weights vector length must match load basis DOFs")
        if np.any(W < 0.0):
            raise ValueError("weights must be non-negative")
        return B.T @ (B * W[:, None])
    if W.ndim == 2:
        if W.shape != (B.shape[0], B.shape[0]):
            raise ValueError("weights matrix shape must match load basis DOFs")
        return B.T @ (W @ B)
    raise ValueError("weights must be either a vector or a square matrix")


def load_basis_correlation_matrix(load_bases: list[PatchLoadBasis] | tuple[PatchLoadBasis, ...]) -> np.ndarray:
    B = assemble_load_basis(load_bases)
    if B.size == 0:
        return np.zeros((0, 0), dtype=float)
    gram = B.T @ B
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denom = np.maximum(np.outer(norms, norms), 1.0e-30)
    return gram / denom


def write_patch_load_basis_tables(
    load_bases: list[PatchLoadBasis] | tuple[PatchLoadBasis, ...],
    patches: list[SurfacePatch] | tuple[SurfacePatch, ...],
    output_dir: str | Path,
) -> None:
    """Write Phase-3 load basis diagnostics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    patch_by_id = {patch.patch_id: patch for patch in patches}
    resultants = load_basis_resultants(load_bases)
    correlations = load_basis_correlation_matrix(load_bases)
    offdiag = correlations - np.eye(correlations.shape[0])
    max_abs_correlation = 0.0 if offdiag.size == 0 else float(np.max(np.abs(offdiag)))
    condition_number = load_basis_condition_number(load_bases)

    stats_lines = [
        "patch_id,basis_type,patch_area,support_nodes,resultant_norm,normal_direction_cosine,"
        "min_node_weight,max_node_weight,node_weight_sum\n"
    ]
    for basis, resultant in zip(load_bases, resultants):
        patch = patch_by_id[basis.patch_id]
        resultant_norm = float(np.linalg.norm(resultant))
        normal_cosine = float(np.dot(resultant, patch.normal) / max(resultant_norm, 1.0e-30))
        node_weights = np.zeros(0, dtype=float) if basis.node_weights is None else basis.node_weights
        stats_lines.append(
            f"{basis.patch_id},{basis.basis_type},{patch.area:.17g},{basis.node_indices.size if basis.node_indices is not None else 0},"
            f"{resultant_norm:.17g},{normal_cosine:.17g},"
            f"{float(np.min(node_weights)) if node_weights.size else 0.0:.17g},"
            f"{float(np.max(node_weights)) if node_weights.size else 0.0:.17g},"
            f"{float(np.sum(node_weights)):.17g}\n"
        )

    condition_lines = [
        "basis_count,condition_number,max_abs_offdiag_correlation\n",
        f"{len(load_bases)},{condition_number:.17g},{max_abs_correlation:.17g}\n",
    ]
    summary = (
        "# Patch Load Basis Summary\n\n"
        f"Basis count: {len(load_bases)}\n"
        f"Condition number: {condition_number:.6g}\n"
        f"Max absolute off-diagonal correlation: {max_abs_correlation:.6g}\n"
    )
    (output_path / "load_basis_stats.csv").write_text("".join(stats_lines), encoding="utf-8")
    (output_path / "basis_condition.csv").write_text("".join(condition_lines), encoding="utf-8")
    (output_path / "load_basis_summary.md").write_text(summary, encoding="utf-8")


def _patch_nodal_area_weights(surface: SurfaceMesh, patch: SurfacePatch) -> dict[int, float]:
    node_area: dict[int, float] = {}
    for tri_id in patch.triangle_indices:
        area_share = float(surface.areas[int(tri_id)]) / 3.0
        for node_id in surface.triangles[int(tri_id)]:
            node_area[int(node_id)] = node_area.get(int(node_id), 0.0) + area_share
    return node_area
