# Patch ILC Claim Gate

Phase 9 gate passed: 1
Claims passed: 6 / 6

## Ablation Set

- A only_low_free_interface_modes: force/error=0.831666, dynamic_dofs=2, alpha_or_patch_dofs=0
- B low_modes_plus_ordinary_patch_static_modes: force/error=0, dynamic_dofs=26, alpha_or_patch_dofs=0
- C low_modes_plus_patch_residual_ilc: force/error=0, dynamic_dofs=2, alpha_or_patch_dofs=24
- D low_modes_plus_sdf_active_patch_residual_ilc: force/error=0.299569, dynamic_dofs=2, alpha_or_patch_dofs=8
- E multi_scale_patch_residual_ilc: force/error=0.269226, dynamic_dofs=2, alpha_or_patch_dofs=32

## Claims

- Claim 1 [PASS]: Low-order modes alone cannot represent local surface contact compliance. Metric quasi_static_low_mode_force_error=0.8316661004869473 threshold > 0.2.
- Claim 2 [PASS]: Patch residual modes recover local quasistatic response under contact loads. Metric static_completeness_and_residual_force_error=2.18033e-17;0 threshold < 1e-6; < 1e-8.
- Claim 3 [PASS]: Patch-level ILC reduces the scale versus node-level ILC. Metric patch_alpha_dofs_vs_node_level_contact_dofs=40 < 69 threshold patch < node.
- Claim 4 [PASS]: SDF active patch detection handles moving contact over patch boundaries. Metric force_jump_gap_jump_unique_requested_patches=0.019911;0.00897824;8 threshold < 0.05; < 0.05; > 1.
- Claim 5 [PASS]: ILC alpha does not enter the main DAE state. Metric dae_dimension_independence_and_dynamic_dof_ratio=1;0.0769231 threshold 1; < 1.
- Claim 6 [PASS]: Multi-scale patch activation improves the accuracy/efficiency balance. Metric multi_error_single_error_active_fraction_runtime_ratio=0.269226;0.299569;0.240964;0.239992 threshold multi <= single; active/runtime < 0.35.
