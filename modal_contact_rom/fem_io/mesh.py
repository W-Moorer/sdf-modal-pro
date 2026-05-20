from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """Minimal 3D finite-element mesh container."""

    nodes: np.ndarray
    elements: np.ndarray
    element_type: str

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        elements = np.asarray(self.elements, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (n_nodes, 3)")
        if elements.ndim != 2:
            raise ValueError("elements must be a 2D integer array")
        if self.element_type not in {"hex8", "tet4"}:
            raise ValueError("element_type must be 'hex8' or 'tet4'")
        expected_width = 8 if self.element_type == "hex8" else 4
        if elements.shape[1] != expected_width:
            raise ValueError(f"{self.element_type} elements must have {expected_width} nodes")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "elements", elements)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_dofs(self) -> int:
        return self.n_nodes * 3

    def dof_indices(self, node_ids: np.ndarray | list[int]) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        return (3 * node_ids[:, None] + np.arange(3, dtype=np.int64)).reshape(-1)


def load_mesh(path: str | Path) -> Mesh:
    """Load the first tetrahedral or hexahedral cell block through meshio."""

    try:
        import meshio
    except ImportError as exc:
        raise ImportError("meshio is required to load external mesh files") from exc

    mesh = meshio.read(path)
    for cell_block in mesh.cells:
        if cell_block.type in {"hexahedron", "hex8"}:
            return Mesh(mesh.points[:, :3], cell_block.data, "hex8")
        if cell_block.type in {"tetra", "tet4"}:
            return Mesh(mesh.points[:, :3], cell_block.data, "tet4")
    raise ValueError("mesh file must contain tetra or hexahedron cells")

