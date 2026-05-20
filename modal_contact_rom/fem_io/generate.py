from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.sparse as sp

from modal_contact_rom.fem_io.mesh import Mesh


@dataclass(frozen=True)
class FEMSystem:
    mesh: Mesh
    K: sp.csr_matrix
    M: sp.csr_matrix
    fixed_nodes: np.ndarray
    fixed_dofs: np.ndarray
    free_dofs: np.ndarray


def cantilever_block(
    nx: int = 4,
    ny: int = 2,
    nz: int = 2,
    length: float = 4.0,
    width: float = 1.0,
    height: float = 1.0,
    density: float = 1.0,
    stiffness_scale: float = 100.0,
    diagonal_shift: float = 1.0e-8,
) -> FEMSystem:
    """Generate a small fixed-left block with a spring-lattice K/M system.

    The spring lattice is not a replacement for production FEM. It is a compact
    positive-definite source of K and M for the first-version algorithm test.
    """

    if min(nx, ny, nz) < 1:
        raise ValueError("nx, ny, nz must all be positive")

    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(-0.5 * width, 0.5 * width, ny + 1)
    zs = np.linspace(-0.5 * height, 0.5 * height, nz + 1)

    def node_id(i: int, j: int, k: int) -> int:
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    nodes = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                nodes.append((xs[i], ys[j], zs[k]))

    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                elements.append(
                    (
                        node_id(i, j, k),
                        node_id(i + 1, j, k),
                        node_id(i + 1, j + 1, k),
                        node_id(i, j + 1, k),
                        node_id(i, j, k + 1),
                        node_id(i + 1, j, k + 1),
                        node_id(i + 1, j + 1, k + 1),
                        node_id(i, j + 1, k + 1),
                    )
                )

    mesh = Mesh(np.asarray(nodes), np.asarray(elements), "hex8")
    cell_volume = length * width * height / float(nx * ny * nz)
    K = _spring_lattice_stiffness(mesh, cell_volume, stiffness_scale)
    if diagonal_shift > 0.0:
        scale = float(np.max(K.diagonal()))
        K = K + sp.eye(mesh.n_dofs, format="csr") * (diagonal_shift * scale)
    M = _lumped_mass(mesh, density=density, cell_volume=cell_volume)

    fixed_nodes = np.flatnonzero(np.isclose(mesh.nodes[:, 0], 0.0))
    fixed_dofs = mesh.dof_indices(fixed_nodes)
    all_dofs = np.arange(mesh.n_dofs, dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=False)
    return FEMSystem(mesh, K.tocsr(), M.tocsr(), fixed_nodes, fixed_dofs, free_dofs)


def _spring_lattice_stiffness(mesh: Mesh, cell_volume: float, stiffness_scale: float) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    pair_ids = list(combinations(range(8), 2))

    for element in mesh.elements:
        for a_local, b_local in pair_ids:
            a = int(element[a_local])
            b = int(element[b_local])
            delta = mesh.nodes[b] - mesh.nodes[a]
            length = float(np.linalg.norm(delta))
            if length <= 0.0:
                continue
            direction = delta / length
            k_local = stiffness_scale * cell_volume / (length * length)
            block = k_local * np.outer(direction, direction)
            dofs_a = 3 * a + np.arange(3)
            dofs_b = 3 * b + np.arange(3)
            for r in range(3):
                for c in range(3):
                    value = float(block[r, c])
                    rows.extend((dofs_a[r], dofs_b[r], dofs_a[r], dofs_b[r]))
                    cols.extend((dofs_a[c], dofs_b[c], dofs_b[c], dofs_a[c]))
                    data.extend((value, value, -value, -value))

    return sp.coo_matrix((data, (rows, cols)), shape=(mesh.n_dofs, mesh.n_dofs)).tocsr()


def _lumped_mass(mesh: Mesh, density: float, cell_volume: float) -> sp.csr_matrix:
    node_mass = np.zeros(mesh.n_nodes, dtype=float)
    element_mass = density * cell_volume
    for element in mesh.elements:
        node_mass[element] += element_mass / element.shape[0]
    diag = np.repeat(node_mass, 3)
    return sp.diags(diag, format="csr")

