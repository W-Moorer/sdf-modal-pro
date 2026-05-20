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
from modal_contact_rom.modal_ilc.patch_residual_modes import (
    PatchResidualBasis,
    PatchResidualBlock,
    build_patch_residual_basis,
    compliance_errors,
    modal_static_contribution,
    solve_static_load_responses,
    static_completeness_errors,
    write_patch_residual_tables,
)
from modal_contact_rom.modal_ilc.reduced_state import ReducedState
from modal_contact_rom.modal_ilc.residual_recovery import recover_residual_deformation, recover_total_deformation
from modal_contact_rom.modal_ilc.sdf_projection import (
    ContactSampleSet,
    PatchILCProjection,
    assemble_sample_lumped_force_vector,
    detect_sdf_contact_samples,
    project_contact_samples_to_patch_ilc,
    total_nodal_force,
    total_nodal_moment,
    write_ilc_projection_tables,
)

__all__ = [
    "ActivePatchSet",
    "ContactSampleSet",
    "PatchILCProjection",
    "PatchLoadBasis",
    "PatchResidualBasis",
    "PatchResidualBlock",
    "ReducedState",
    "assemble_load_basis",
    "assemble_sample_lumped_force_vector",
    "build_patch_residual_basis",
    "build_normal_patch_load_basis",
    "build_normal_patch_load_bases",
    "compliance_errors",
    "detect_sdf_contact_samples",
    "load_basis_condition_number",
    "load_basis_correlation_matrix",
    "load_basis_gram",
    "load_basis_resultants",
    "modal_static_contribution",
    "project_contact_force_to_ilc",
    "project_contact_samples_to_patch_ilc",
    "recover_residual_deformation",
    "recover_total_deformation",
    "solve_static_load_responses",
    "static_completeness_errors",
    "total_nodal_force",
    "total_nodal_moment",
    "write_ilc_projection_tables",
    "write_patch_load_basis_tables",
    "write_patch_residual_tables",
]
