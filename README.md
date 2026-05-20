# Modal Contact ROM

This repository contains the first-version prototype described in `aim.md`.
It validates this claim:

```text
Adding surface patch contact modes improves local surface compliance compared
with a basis that only contains low-frequency vibration modes.
```

The first version is intentionally standalone. It uses Python, NumPy, SciPy,
and optional meshio input. The default example generates a small cantilever
block mesh, builds a positive-definite spring-lattice K/M system, extracts the
outer surface, partitions it into patches, solves one static contact mode per
patch, mass-orthonormalizes the combined basis, and compares reduced static
compliance errors.

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

## Modules

```text
modal_contact_rom/
  fem_io/                mesh data, meshio loading, generated block K/M
  modal_basis/           constrained generalized eigenmodes
  surface_patch/         outer surface extraction and patch partitioning
  contact_modes/         patch normal loads and static flexibility modes
  reduced_dynamics/      mass orthonormalization and reduced static solve
  sdf_query/             simple triangle-mesh signed distance prototype
  adaptive_activation/   nearest patch activation helper
  validation/            compliance error validation
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
Python full-order FEM, adaptive ROM, and WSL CalculiX direct full-order FEM:

```powershell
python examples/compare_three_way_dynamics.py
```
