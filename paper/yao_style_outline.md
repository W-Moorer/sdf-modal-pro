# Paper Outline Following Yao et al. Reference 9

## 1. Analysis of the Yao et al. Paper Format

Reference 9 is structured as a mechanics-method paper rather than a purely algorithmic paper. Its main writing pattern is:

1. Start from an engineering contact problem where full FEM is accurate but too expensive.
2. Review existing MOR/CMS/contact-interface treatments and identify one central bottleneck: local contact accuracy requires many interface DOFs.
3. Introduce one reduced formulation that keeps local contact accuracy while preventing the DAE dimension from growing.
4. Derive the method in a layered order: kinematics, modal convergence criterion, mode-set construction, residual-mode calculation, equations of motion, efficient coordinate solution, contact algorithm.
5. Validate progressively: static completeness, truncation criterion, contact algorithm, dynamic comparison with full FEM or prior methods, then a larger application that full FEM cannot conveniently cover.
6. Close with concise claims on accuracy, efficiency, and generalization to other moving-contact systems.

The paper's section flow is:

- Abstract: problem, FEM limitation, proposed free-interface CMS, residual modes, interface load coordinates, validation.
- 1 Introduction: moving-contact motivation, FEM cost, modal-only limitation, static correction modes, SMS/ALE/PMOR comparison, contribution and section roadmap.
- 2 Free-interface CMS of moving contact multibodies:
  - 2.1 FFR kinematics for moving contact bodies.
  - 2.2 convergence criterion separating low-frequency dynamic modes and high-frequency quasi-static modes.
  - 2.3 mode set: kept normal modes plus residual modes driven by interface loads.
  - 2.4 residual-mode calculation by static attachment response minus retained modal static contribution.
- 3 Simplification of equations of motion: coordinates, energy terms, mass/stiffness simplifications, final DAE form.
- 4 Efficient solution for ILCs and computational flowchart: direct update of interface load coordinates and implementation workflow.
- 5 Contact algorithm: candidate contact pair detection, boundary detection, force/equivalent nodal load calculation.
- 6 Numerical examples:
  - static completeness verification,
  - frequency truncation criterion,
  - contact algorithm verification,
  - dynamic verification against FEM/ALE,
  - high-speed gear traveling-wave application.
- 7 Conclusions: restate method, numerical evidence, efficiency gain, broader applications.

The reusable template is therefore: theory first, algorithm second, contact treatment third, validation matrix last.

## 2. Proposed Paper Title

SDF-Driven Patch Residual Interface Load Coordinates for Adaptive Modal Contact Dynamics

## 3. Abstract Outline

Paragraph role: problem and gap.

- Moving contact in flexible multibody systems needs local surface compliance and high-gradient contact response.
- Full FEM captures these effects but is too expensive for repeated contact-state changes.
- Standard low-mode ROMs miss local compliance; ordinary static modes recover it but enlarge the dynamic basis or switch modes.

Paragraph role: method.

- Propose a patch residual interface load coordinate method.
- Low free-interface modes represent global deformation.
- Patch residual modes recover local quasi-static contact response.
- SDF-driven active patch detection maps moving contact samples to patch ILCs.
- Multi-scale patch activation smooths contact transitions without adding alpha to the main DAE state.

Paragraph role: evidence.

- Static completeness, compliance recovery, quasi-static indentation, moving contact smoothness, small full-FEM dynamics, and large-case performance are validated.
- Phase 9 ablation gates verify A-E variants and six paper claims.

## 4. Section-by-Section Outline

### 1 Introduction

1. Moving-contact flexible bodies need both global dynamics and local surface response.
2. Full FEM is accurate but too costly when the contact area moves over a fine surface.
3. Low-mode ROMs are efficient but cannot represent local contact compliance.
4. Ordinary static correction modes restore local response but increase the dynamic coordinate set or require switching.
5. Existing alternatives such as static mode switching, ALE contact reduction, and parametric MOR reduce some costs but still introduce basis-update, smoothness, or parameterization issues.
6. This paper proposes patch residual ILCs driven by SDF contact samples, keeping the dynamic DAE dimension independent of active contact nodes.
7. Contributions:
   - patch residual modes for local contact compliance,
   - alpha/ILC layer outside the dynamic state,
   - SDF-to-active-patch projection,
   - online residual coupling,
   - multi-scale patch activation with overlap and warm start,
   - validation and ablation claim gates.
8. Paper roadmap.

### 2 Patch Residual ILC Formulation for Moving Contact Flexible Bodies

#### 2.1 Flexible body kinematics and reduced coordinates

- Define full displacement field and retained modal coordinates.
- Separate the main dynamic state from contact/interface load coordinates.
- State the core design principle: alpha is an interface/load coordinate, not a DAE state variable.

Expected figure:

- `method_pipeline.png`: FEM mesh -> surface patches -> patch load basis -> residual ILC -> SDF active set -> claim gate.

#### 2.2 Convergence criterion for local contact response

- Follow Yao's logic: low-frequency modes carry global dynamics; omitted high-frequency modes contribute quasi-statically under localized contact loads.
- Explain why local contact compliance cannot converge with low modes alone.
- Define the target static attachment response for each patch load.

Expected table:

- `static_completeness.csv`: per-patch static reconstruction error.

#### 2.3 Patch load basis and interface load coordinates

- Partition the surface into patches.
- Build area-weighted normal patch load bases.
- Define active patch set and alpha vector.
- Explain how patch-level ILCs reduce node-level interface scale.

Expected evidence:

- Claim 3: multi-scale patch alpha DOFs = 40 versus node-level contact DOFs = 69.

#### 2.4 Patch residual modes

- Define full static response `G_B`.
- Define retained modal static contribution.
- Define residual modes as the difference between full static attachment response and modal static contribution.
- Explain static completeness: modal static contribution plus residual response reconstructs full static attachment response.

Expected table:

- `compliance_error.csv`: low-mode compliance error, residual ILC error, compressed residual error.

### 3 Equations of Motion and DAE-Dimension Invariance

#### 3.1 Dynamic state without alpha

- Define reduced displacement/velocity/acceleration in retained modal coordinates.
- State that alpha contributes only through recovered residual deformation/contact correction.

#### 3.2 Online residual coupling

- Show contact residual projection into patch alpha.
- Show how residual deformation is recovered from active residual blocks.
- Explain frozen-alpha or tangent-style iteration used in the current implementation.

#### 3.3 DAE dimension argument

- Compare:
  - low modes only,
  - ordinary patch static modes as dynamic modes,
  - patch residual ILC.
- Emphasize that ordinary static modes increase dynamic DOFs, while residual ILC keeps the main DAE dimension unchanged.

Expected evidence:

- Claim 5: DAE independence flag = 1; residual dynamic DOF ratio vs ordinary static modes = 0.0769231.

### 4 Efficient Active ILC Solution and Multi-Scale Patch Activation

#### 4.1 SDF contact sample to patch ILC projection

- Detect contact samples with SDF queries.
- Map closest surface triangles to primary patches.
- Aggregate normal contact forces into patch alpha.
- Verify total force and normal-force consistency.

#### 4.2 Adaptive active patch management

- Current contact patches must activate.
- Neighbor patches activate for one-ring smoothness.
- Hysteresis and deactivation delay prevent discontinuous active-set changes.
- Alpha warm start smooths transitions.

#### 4.3 Multi-scale patch activation

- Coarse patches provide global safeguard.
- Medium patches cover current contact neighborhoods.
- Fine overlap patches activate when projection error/contact force gradient is large.
- Partition-of-unity weights distribute force across overlapping patches.

Expected figure/table:

- `patch_hierarchy.png`
- `active_patch_history.png`
- `active_patch_history.csv`

### 5 Contact Treatment

#### 5.1 SDF-based contact detection

- Define query points, closest surface triangles, gaps, normals, and penetration.
- Explain why SDF/BVH-style queries are suitable for arbitrary exterior surfaces.

#### 5.2 Force projection and penalty calibration

- Convert pressure/penetration to normal nodal or patch loads.
- Use area weighting to align penalty units with pressure-overclosure contact.
- State relation to CalculiX-aligned validation from earlier project phases.

#### 5.3 Contact boundary smoothness

- Explain boundary-crossing force jump, gap jump, and alpha jump metrics.
- Use moving contact over patch boundaries as the main smoothness test.

Expected evidence:

- moving force jump = 0.019911
- moving gap jump = 0.00897824
- energy drift excess ratio = 0.954554

### 6 Numerical Examples and Claim-Gated Validation

#### 6.1 Static completeness verification

Purpose: prove residual modes recover the exact static attachment response.

Metrics:

- maximum static completeness error,
- residual compliance error,
- compressed residual compliance error.

Expected results:

- max completeness error = 2.180325e-17
- compressed residual compliance error = 2.456128e-02

#### 6.2 Quasi-static indentation verification

Purpose: prove local contact stiffness is recovered.

Models:

- low modes only,
- low modes plus ordinary patch static modes,
- low modes plus patch residual ILC.

Metrics:

- force-displacement curve,
- local indentation,
- patch compliance,
- gap,
- contact region size,
- active DOFs.

Expected figure/table:

- `force_displacement_curve.png`
- `contact_force_error.csv`

#### 6.3 Moving contact verification

Purpose: prove SDF active patch detection works across patch boundaries.

Metrics:

- contact force continuity,
- gap continuity,
- active patch count,
- runtime ratio,
- energy drift.

Expected results:

- force jump < 5%;
- gap jump < 5%;
- adaptive energy drift not higher than full active-patch baseline.

#### 6.4 Small full-FEM dynamic comparison

Purpose: provide a small full-FEM time-history reference without relying on large-scale full FEM.

Metrics:

- displacement RMSE,
- velocity RMSE,
- contact force peak error,
- contact impulse error,
- measured wall-time runtime ratio,
- solver-dimension speedup.

Expected results:

- displacement RMSE = 6.448974e-06
- velocity RMSE = 6.706009e-04
- peak force error = 0
- impulse error = 0
- solver-dimension speedup = 6.292217x

#### 6.5 Large performance verification

Purpose: show scaling behavior without running large full-FEM time history.

Compare:

- baseline modal-SDF,
- patch residual ILC,
- node-level ILC estimate,
- coarse-only patch,
- multi-scale patch.

Expected table/figure:

- `runtime_scaling.csv`
- `runtime_scaling.png`

#### 6.6 Ablation study and claim gate

Required ablations:

- A: only low free-interface modes.
- B: low modes + ordinary patch static modes as dynamic modes.
- C: low modes + patch residual ILC.
- D: low modes + SDF-driven active patch residual ILC.
- E: multi-scale patch residual ILC.

Required claims:

- Claim 1: low modes alone cannot represent local contact compliance.
- Claim 2: patch residual modes recover local quasi-static contact response.
- Claim 3: patch-level ILC reduces node-level ILC scale.
- Claim 4: SDF active patch detection handles moving contact across arbitrary surfaces.
- Claim 5: ILC does not enter the main DAE variables.
- Claim 6: multi-scale activation improves the accuracy/efficiency balance.

Expected outputs:

- `ablation_summary.csv`
- `claim_gate.csv`
- `patch_ilc_claims.md`

### 7 Conclusions

Paragraph role: method recap.

- Patch residual ILC separates global dynamics from local contact compliance.
- SDF active patch projection enables moving contact over arbitrary exterior surfaces.
- Multi-scale activation improves patch-level accuracy/efficiency.

Paragraph role: evidence recap.

- Static completeness and compliance recovery are verified.
- Moving contact has bounded force and gap jumps.
- Small full-FEM dynamics and large-case performance proxies support the method.
- Phase 9 claim gate passes all six claims.

Paragraph role: limitations and future work.

- Current method focuses on frictionless normal contact.
- Future work should extend to frictional tangential basis, nonlinear material/contact laws, stronger external FEM benchmarks, and larger industrial geometry.

## 5. Figure and Table Plan

### Figures

1. `method_pipeline.png`: method overview.
2. `patch_hierarchy.png`: coarse/medium/fine patch hierarchy.
3. `force_displacement_curve.png`: quasi-static indentation ablation.
4. `active_patch_history.png`: moving contact active patch count and force.
5. `runtime_scaling.png`: runtime proxy comparison.

### Tables

1. `static_completeness.csv`: patch static reconstruction.
2. `compliance_error.csv`: compliance recovery.
3. `contact_force_error.csv`: indentation and dynamic force errors.
4. `runtime_scaling.csv`: scalability comparison.
5. `ablation_summary.csv`: A-E ablation matrix.
6. `claim_gate.csv`: six claim pass/fail evidence.

## 6. Claim-Evidence Map

| Claim | Evidence | Status |
| --- | --- | --- |
| Low modes alone cannot represent local contact compliance. | Quasi-static low-mode force error = 0.831666. | Supported |
| Patch residual modes recover local quasi-static response. | Static completeness error = 2.180325e-17; residual force error = 0. | Supported |
| Patch-level ILC reduces node-level scale. | Patch alpha DOFs = 40 vs node-level contact DOFs = 69. | Supported |
| SDF active patch detection handles moving contact. | Force jump = 0.019911; gap jump = 0.00897824; requested patches = 8. | Supported |
| ILC does not enter the main DAE state. | DAE independence flag = 1; dynamic DOF ratio vs ordinary modes = 0.0769231. | Supported |
| Multi-scale activation improves accuracy/efficiency balance. | Multi-scale projection error = 0.269226 vs single-scale = 0.299569; active fraction = 0.240964. | Supported |

## 7. Self-Review Checklist

- Does the introduction end with a clear bottleneck: local contact response without DAE growth?
- Are residual modes defined as full static attachment response minus retained modal static contribution?
- Is alpha consistently described as an interface load coordinate rather than a dynamic state?
- Does every abstract/introduction claim map to a Phase 8 or Phase 9 output?
- Are all figures and tables necessary for the six claims?
- Are limitations explicit enough: frictionless normal contact, prototype-scale dynamic FEM comparison, future industrial validation?
