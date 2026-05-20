from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from modal_contact_rom.fem_io.generate import FEMSystem, cantilever_block
from modal_contact_rom.fem_io.mesh import Mesh


def assemble_sfc_fem_system(
    mesh: Mesh,
    *,
    young_modulus: float = 1000.0,
    poisson_ratio: float = 0.3,
    density: float = 1.0,
    mass_kind: str = "consistent",
    fixed_nodes: np.ndarray | None = None,
    diagonal_shift: float = 0.0,
) -> FEMSystem:
    """Assemble a ``FEMSystem`` with the vendored SFC finite-element kernels."""

    from sfc.fem import DeformableBody, assemble_mass_matrix, assemble_stiffness_matrix
    from sfc.mesh import VolumeMesh

    volume_mesh = VolumeMesh(mesh.nodes, mesh.elements, element_type=mesh.element_type)
    body = DeformableBody(volume_mesh, {"E": float(young_modulus), "nu": float(poisson_ratio)}, density=float(density))
    K = assemble_stiffness_matrix(body).tocsr()
    M = assemble_mass_matrix(body, kind=mass_kind).tocsr()
    if diagonal_shift > 0.0:
        scale = float(np.max(K.diagonal())) if K.nnz else 1.0
        K = K + sp.eye(mesh.n_dofs, format="csr") * (float(diagonal_shift) * scale)

    if fixed_nodes is None:
        fixed = np.flatnonzero(np.isclose(mesh.nodes[:, 0], np.min(mesh.nodes[:, 0])))
    else:
        fixed = np.asarray(fixed_nodes, dtype=np.int64).reshape(-1)
    fixed_dofs = mesh.dof_indices(fixed)
    all_dofs = np.arange(mesh.n_dofs, dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=False)
    return FEMSystem(mesh, K, M, fixed, fixed_dofs, free_dofs)


def sfc_cantilever_block(
    nx: int = 4,
    ny: int = 2,
    nz: int = 2,
    length: float = 4.0,
    width: float = 1.0,
    height: float = 1.0,
    young_modulus: float = 1000.0,
    poisson_ratio: float = 0.3,
    density: float = 1.0,
    mass_kind: str = "consistent",
    diagonal_shift: float = 0.0,
) -> FEMSystem:
    """Generate the project cantilever mesh and assemble K/M through SFC."""

    seed = cantilever_block(
        nx=nx,
        ny=ny,
        nz=nz,
        length=length,
        width=width,
        height=height,
        density=density,
        stiffness_scale=1.0,
        diagonal_shift=0.0,
    )
    return assemble_sfc_fem_system(
        seed.mesh,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        mass_kind=mass_kind,
        fixed_nodes=seed.fixed_nodes,
        diagonal_shift=diagonal_shift,
    )
