"""Patch-ILC residual correction layer."""

from modal_contact_rom.modal_ilc.ilc_projection import ActivePatchSet, project_contact_force_to_ilc
from modal_contact_rom.modal_ilc.patch_load_basis import (
    PatchLoadBasis,
    assemble_load_basis,
    build_normal_patch_load_basis,
    build_normal_patch_load_bases,
    load_basis_condition_number,
    load_basis_correlation_matrix,
    load_basis_gram,
    load_basis_resultants,
    write_patch_load_basis_tables,
)
from modal_contact_rom.modal_ilc.patch_residual_modes import PatchResidualBlock
from modal_contact_rom.modal_ilc.reduced_state import ReducedState
from modal_contact_rom.modal_ilc.residual_recovery import recover_residual_deformation, recover_total_deformation

__all__ = [
    "ActivePatchSet",
    "PatchLoadBasis",
    "PatchResidualBlock",
    "ReducedState",
    "assemble_load_basis",
    "build_normal_patch_load_basis",
    "build_normal_patch_load_bases",
    "load_basis_condition_number",
    "load_basis_correlation_matrix",
    "load_basis_gram",
    "load_basis_resultants",
    "project_contact_force_to_ilc",
    "recover_residual_deformation",
    "recover_total_deformation",
    "write_patch_load_basis_tables",
]
