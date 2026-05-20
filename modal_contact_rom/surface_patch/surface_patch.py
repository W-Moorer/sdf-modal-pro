from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modal_contact_rom.surface_patch.extract import SurfaceMesh


@dataclass(frozen=True)
class SurfaceSamples:
    """Triangle-quadrature samples used to drive patch hierarchy construction."""

    triangle_indices: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    areas: np.ndarray

    @property
    def count(self) -> int:
        return int(self.triangle_indices.size)


def build_surface_samples(surface: SurfaceMesh) -> SurfaceSamples:
    """Use one area-weighted sample at each surface triangle centroid."""

    triangle_indices = np.arange(surface.triangles.shape[0], dtype=np.int64)
    return SurfaceSamples(
        triangle_indices=triangle_indices,
        points=np.asarray(surface.centroids, dtype=float),
        normals=np.asarray(surface.normals, dtype=float),
        areas=np.asarray(surface.areas, dtype=float),
    )
