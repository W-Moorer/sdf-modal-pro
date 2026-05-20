from __future__ import annotations

import numpy as np

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.fem_io.mesh import Mesh
from modal_contact_rom.modal_ilc import (
    assemble_load_basis,
    build_normal_patch_load_bases,
    load_basis_condition_number,
    load_basis_correlation_matrix,
    load_basis_resultants,
    write_patch_load_basis_tables,
)
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def test_every_patch_has_unit_resultant_normal_load_basis() -> None:
    fem = cantilever_block(nx=3, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    patches = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium").patches
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, patches)
    resultants = load_basis_resultants(load_bases)

    assert len(load_bases) == len(patches)
    for basis, patch, resultant in zip(load_bases, patches, resultants):
        assert basis.basis_type == "normal"
        assert basis.load_dimension == 1
        assert basis.node_weights is not None
        assert basis.node_indices is not None
        assert np.all(basis.node_weights >= 0.0)
        np.testing.assert_allclose(np.sum(basis.node_weights), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(np.linalg.norm(resultant), 1.0, atol=1.0e-12)
        assert float(np.dot(resultant, patch.normal)) > 0.99


def test_normal_patch_load_basis_is_supported_only_on_own_patch_nodes() -> None:
    fem = cantilever_block(nx=3, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    patches = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium").patches
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, patches)

    for basis, patch in zip(load_bases, patches):
        nodal_vectors = basis.B_j.reshape(fem.mesh.n_nodes, 3)
        support_nodes = np.flatnonzero(np.linalg.norm(nodal_vectors, axis=1) > 1.0e-14)
        assert set(int(node_id) for node_id in support_nodes) == set(int(node_id) for node_id in basis.node_indices)
        assert set(int(node_id) for node_id in support_nodes).issubset(set(int(node_id) for node_id in patch.node_indices))
        outside = np.setdiff1d(np.arange(fem.mesh.n_nodes), patch.node_indices, assume_unique=False)
        np.testing.assert_allclose(nodal_vectors[outside], 0.0, atol=1.0e-14)


def test_active_patch_load_basis_conditioning_and_adjacency_correlation_are_acceptable() -> None:
    fem = cantilever_block(nx=3, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    level = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, level.patches)

    B = assemble_load_basis(load_bases)
    correlation = load_basis_correlation_matrix(load_bases)
    patch_id_to_column = {basis.patch_id: column for column, basis in enumerate(load_bases)}
    max_adjacent_correlation = max(
        abs(correlation[patch_id_to_column[patch_id], patch_id_to_column[neighbor_id]])
        for patch_id, neighbors in level.adjacency.items()
        for neighbor_id in neighbors
    )

    assert B.shape == (fem.mesh.n_dofs, len(level.patches))
    assert load_basis_condition_number(load_bases) < 1.0e8
    assert max_adjacent_correlation < 0.99


def test_irregular_surface_load_basis_writes_phase3_diagnostics(tmp_path) -> None:
    base = cantilever_block(nx=2, ny=1, nz=1).mesh
    nodes = base.nodes.copy()
    nodes[:, 1] += 0.08 * np.sin(1.7 * nodes[:, 0] + 0.3 * nodes[:, 2])
    nodes[:, 2] += 0.05 * np.cos(1.1 * nodes[:, 0] - 0.4 * nodes[:, 1])
    mesh = Mesh(nodes=nodes, elements=base.elements, element_type=base.element_type)
    surface = extract_surface(mesh)
    patches = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 1)).level("medium").patches
    load_bases = build_normal_patch_load_bases(mesh, surface, patches)

    assert load_basis_condition_number(load_bases) < 1.0e8
    write_patch_load_basis_tables(load_bases, patches, tmp_path)

    assert (tmp_path / "load_basis_stats.csv").read_text(encoding="utf-8").startswith("patch_id,basis_type")
    assert "condition_number" in (tmp_path / "basis_condition.csv").read_text(encoding="utf-8")
    assert "Patch Load Basis Summary" in (tmp_path / "load_basis_summary.md").read_text(encoding="utf-8")
