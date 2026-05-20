from __future__ import annotations

import numpy as np

from modal_contact_rom.surface_patch.extract import SurfacePatch


def nearest_patch_ids(
    point: np.ndarray,
    patches: list[SurfacePatch],
    max_count: int = 1,
    radius: float | None = None,
) -> list[int]:
    """Return patch ids nearest to a contact point."""

    if max_count < 1:
        raise ValueError("max_count must be positive")
    point = np.asarray(point, dtype=float)
    distances = []
    for patch in patches:
        distance = float(np.linalg.norm(point - patch.centroid))
        if radius is None or distance <= radius:
            distances.append((distance, patch.patch_id))
    distances.sort()
    return [patch_id for _, patch_id in distances[:max_count]]

