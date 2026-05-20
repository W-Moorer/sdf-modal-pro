from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicAccuracySummary:
    relative_displacement_l2: float
    final_relative_displacement_l2: float
    relative_velocity_l2: float
    relative_force_l2: float
    gap_rmse: float
    max_gap_error: float
    max_energy_balance_error_difference: float


@dataclass(frozen=True)
class ForcedAccuracySummary:
    relative_displacement_l2: float
    final_relative_displacement_l2: float
    relative_velocity_l2: float
    max_energy_balance_error_difference: float


@dataclass(frozen=True)
class CalculixObservedDisplacementSummary:
    relative_observed_displacement_l2: float
    max_observed_displacement_error: float
    sample_count: int
    node_count: int


def compare_dynamic_results(full_result, rom_result) -> DynamicAccuracySummary:
    """Compare ROM dynamics against a full-order FEM reference trajectory."""

    if rom_result.displacement_history is None or rom_result.velocity_history is None:
        raise ValueError("ROM result must contain displacement and velocity histories")
    full_u = np.asarray(full_result.displacement_history, dtype=float)
    rom_u = np.asarray(rom_result.displacement_history, dtype=float)
    full_v = np.asarray(full_result.velocity_history, dtype=float)
    rom_v = np.asarray(rom_result.velocity_history, dtype=float)
    if full_u.shape != rom_u.shape:
        raise ValueError(f"displacement history shape mismatch: {full_u.shape} != {rom_u.shape}")
    if full_v.shape != rom_v.shape:
        raise ValueError(f"velocity history shape mismatch: {full_v.shape} != {rom_v.shape}")

    full_force = np.asarray([step.normal_force for step in full_result.steps], dtype=float)
    rom_force = np.asarray([step.normal_force for step in rom_result.steps], dtype=float)
    full_gap = np.asarray([step.min_gap for step in full_result.steps], dtype=float)
    rom_gap = np.asarray([step.min_gap for step in rom_result.steps], dtype=float)
    full_energy_error = np.asarray([step.energy_balance_error for step in full_result.steps], dtype=float)
    rom_energy_error = np.asarray([step.energy_balance_error for step in rom_result.steps], dtype=float)

    return DynamicAccuracySummary(
        relative_displacement_l2=_relative_l2(full_u, rom_u),
        final_relative_displacement_l2=_relative_l2(full_u[-1], rom_u[-1]),
        relative_velocity_l2=_relative_l2(full_v, rom_v),
        relative_force_l2=_relative_l2(full_force, rom_force),
        gap_rmse=float(np.sqrt(np.mean((full_gap - rom_gap) ** 2))),
        max_gap_error=float(np.max(np.abs(full_gap - rom_gap))),
        max_energy_balance_error_difference=float(np.max(np.abs(full_energy_error - rom_energy_error))),
    )


def compare_forced_results(reference_result, candidate_result) -> ForcedAccuracySummary:
    """Compare two prescribed-load dynamic trajectories."""

    reference_u = np.asarray(reference_result.displacement_history, dtype=float)
    candidate_u = np.asarray(candidate_result.displacement_history, dtype=float)
    reference_v = np.asarray(reference_result.velocity_history, dtype=float)
    candidate_v = np.asarray(candidate_result.velocity_history, dtype=float)
    reference_energy = np.asarray([step.energy_balance_error for step in reference_result.steps], dtype=float)
    candidate_energy = np.asarray([step.energy_balance_error for step in candidate_result.steps], dtype=float)
    return ForcedAccuracySummary(
        relative_displacement_l2=_relative_l2(reference_u, candidate_u),
        final_relative_displacement_l2=_relative_l2(reference_u[-1], candidate_u[-1]),
        relative_velocity_l2=_relative_l2(reference_v, candidate_v),
        max_energy_balance_error_difference=float(np.max(np.abs(reference_energy - candidate_energy))),
    )


def compare_to_calculix_observed_displacements(result, node_history, observed_nodes: np.ndarray) -> CalculixObservedDisplacementSummary:
    """Compare a Python trajectory to CalculiX *NODE PRINT displacement output."""

    observed_nodes = np.asarray(observed_nodes, dtype=np.int64)
    python_blocks: list[np.ndarray] = []
    calculix_blocks: list[np.ndarray] = []
    for node_id in observed_nodes:
        if int(node_id) not in node_history.displacements:
            continue
        python_values = result.displacement_history[:, 3 * int(node_id) : 3 * int(node_id) + 3]
        calculix_values = node_history.displacements[int(node_id)]
        count = min(python_values.shape[0], calculix_values.shape[0])
        if count == 0:
            continue
        python_blocks.append(python_values[:count])
        calculix_blocks.append(calculix_values[:count])
    if not python_blocks:
        raise ValueError("none of the requested observed nodes were present in CalculiX output")
    python_stack = np.vstack(python_blocks)
    calculix_stack = np.vstack(calculix_blocks)
    return CalculixObservedDisplacementSummary(
        relative_observed_displacement_l2=_relative_l2(calculix_stack, python_stack),
        max_observed_displacement_error=float(np.max(np.abs(calculix_stack - python_stack))),
        sample_count=int(python_stack.shape[0]),
        node_count=len(python_blocks),
    )


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = float(np.linalg.norm(reference - candidate))
    denominator = max(float(np.linalg.norm(reference)), 1.0e-30)
    return numerator / denominator
