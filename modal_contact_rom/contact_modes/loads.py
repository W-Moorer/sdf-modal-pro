from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.surface_patch.extract import SurfaceMesh, SurfacePatch


@dataclass(frozen=True)
class PatchLoad:
    patch_id: int
    vector: np.ndarray
    area: float
    resultant: np.ndarray


def build_patch_normal_loads(
    mesh: Mesh,
    surface: SurfaceMesh,
    patches: list[SurfacePatch],
    normalize_resultant: bool = True,
) -> list[PatchLoad]:
    """Build one distributed normal load vector per surface patch."""

    loads: list[PatchLoad] = []
    for patch in patches:
        b = np.zeros(mesh.n_dofs, dtype=float)
        resultant = np.zeros(3, dtype=float)
        for tri_id in patch.triangle_indices:
            triangle = surface.triangles[tri_id]
            nodal_force = surface.normals[tri_id] * surface.areas[tri_id] / 3.0
            resultant += surface.normals[tri_id] * surface.areas[tri_id]
            for node_id in triangle:
                start = 3 * int(node_id)
                b[start : start + 3] += nodal_force
        if normalize_resultant:
            scale = float(np.linalg.norm(resultant))
            if scale > 0.0:
                b /= scale
                resultant /= scale
        loads.append(PatchLoad(patch.patch_id, b, patch.area, resultant))
    return loads

