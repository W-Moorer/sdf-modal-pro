# Modal Contact ROM

This repository contains the first-version prototype described in `aim.md`.
It validates this claim:

```text
Adding surface patch contact modes improves local surface compliance compared
with a basis that only contains low-frequency vibration modes.
```

The first version uses Python, NumPy, SciPy, and optional meshio input. The
default example generates a small cantilever block mesh, builds a
positive-definite spring-lattice K/M system, extracts the outer surface,
partitions it into patches, solves one static contact mode per patch,
mass-orthonormalizes the combined basis, and compares reduced static
compliance errors. Newer dynamic validation paths can assemble true HEX8 FEM
matrices through the vendored `sfc` package copied from `sdf-fem-pro/src/sfc`.

## Run

```powershell
python examples/run_first_version.py
```

Expected result: `combined` has much lower compliance and energy errors than
`low_modes`.

To generate validation plots and surface diagnostic clouds:

```powershell
python examples/visualize_first_version.py
```

## Phase 2 Patch Hierarchy

The second roadmap phase adds an explicit outer-surface hierarchy:
volume mesh boundary faces are converted to surface samples, then grouped into
coarse and medium patch levels. Each level stores triangle-to-patch assignment
weights, patch adjacency, coverage, and area diagnostics. The overlap medium
level uses partition-of-unity weights so every surface triangle still sums to a
unit assignment.

```powershell
python examples/run_patch_hierarchy.py
```

## Phase 3 Patch Load Basis

The third roadmap phase builds one frictionless normal load basis per surface
patch. Nodal loads are area-weighted within the patch, aligned with the patch
average normal, and normalized to unit resultant force. Diagnostics check
resultant direction, non-negative nodal weights, support locality, basis
conditioning, and cross-patch correlation.

```powershell
python examples/run_patch_load_basis.py
```

## Phase 4 Patch Residual Modes

The fourth roadmap phase builds offline residual modes for the patch load
basis. For each patch load column it solves the constrained static attachment
problem `K G_B = B`, subtracts the retained modal static contribution
`Psi K_kk^-1 Psi.T B`, and stores the remaining local correction as
`Phi_B`. The reconstruction `Psi K_kk^-1 Psi.T B alpha + Phi_B alpha`
matches the full static attachment response while keeping `alpha` outside the
dynamic state.

```powershell
python examples/run_patch_residual_modes.py
```

## Phase 5 SDF To Patch ILC

The fifth roadmap phase connects SDF contact samples to active patch ILCs.
Signed-distance contact samples are mapped through their closest surface
triangle into the active patch level, normal contact forces are aggregated per
patch, and the active load basis reconstructs `B_A alpha_A`. Diagnostics report
contact-to-patch mapping, total force conservation, normal force conservation,
and moment error.

```powershell
python examples/run_ilc_projection.py
```

## Modules

```text
modal_contact_rom/
  fem_io/                mesh data, meshio loading, generated block K/M
  modal_basis/           constrained generalized eigenmodes
  surface_patch/         outer surface extraction, samples, patch hierarchy
  contact_modes/         patch normal loads and static flexibility modes
  modal_ilc/             reduced state, patch ILCs, load basis, recovery
  reduced_dynamics/      mass orthonormalization and reduced static solve
  sdf_query/             simple triangle-mesh signed distance prototype
  adaptive_activation/   nearest patch activation helper
  validation/            compliance error validation
sfc/                     vendored SFC FEM/SDF/contact kernels
```

## Test

```powershell
python -m pytest tests
```

## CalculiX Through WSL

The project can call CalculiX from Windows through WSL. The example below
generates a C3D8 cantilever input deck, runs `ccx` in WSL with
`*FREQUENCY, SOLVER=MATRIXSTORAGE`, reads `job.sti`, `job.mas`, and `job.dof`,
then feeds the CalculiX K/M matrices into the same patch-ROM validation chain.

```powershell
python examples/run_calculix_matrix_storage.py
```

## Second Version Dynamics

The second-version prototype adds a Craig-Bampton connection basis, modal
coordinate time integration, rigid-sphere contact, contact-force projection,
online patch-mode activation, and force/gap/energy diagnostics.

```powershell
python examples/run_second_version.py
python examples/visualize_second_version.py
```

For an external-solver accuracy check, run a three-way comparison between
Python full-order FEM, projected ROM, and WSL CalculiX direct full-order FEM:

```powershell
python examples/compare_three_way_dynamics.py
```

The nonlinear contact three-way check first aligns SFC-assembled Python
full-order FEM
against a CalculiX `*CONTACT PAIR` full-order run, then compares CalculiX
full-order contact, Python full-order contact, and adaptive ROM contact. The
CalculiX input uses surface-to-surface contact, and the Python full-order path
uses the same pressure-overclosure stiffness with surface quadrature, contact
tangent, and implicit Newton iterations.

## Third Version Adaptive Modal Library

The third-version path replaces the older explicit nodal adaptive contact
prototype with `AdaptiveCalculixAlignedROMContactSimulator`. It keeps the same
surface-quadrature contact residual and tangent as the aligned full-order FEM,
but updates the reduced basis online:

- contact quadrature points activate the nearest normal-aligned surface patch
  modes;
- the full-order equilibrium residual is projected onto inactive patch modes,
  activating missing modal directions when their virtual-work score is large;
- when the active set changes, the time step is retried from the last accepted
  state with the updated reduced contact Jacobian.

This gives an adaptive modal flexible-body validation path that can be compared
directly against both SFC full-order FEM and CalculiX nonlinear contact:

```powershell
python examples/compare_three_way_contact_dynamics.py
```
