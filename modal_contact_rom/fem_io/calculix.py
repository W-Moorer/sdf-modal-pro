from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
import re
import shlex
import subprocess

import numpy as np
import scipy.sparse as sp

from modal_contact_rom.fem_io.generate import FEMSystem
from modal_contact_rom.fem_io.mesh import Mesh


@dataclass(frozen=True)
class CalculixRunResult:
    job_name: str
    workdir: Path
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CalculixDofMap:
    node_ids: np.ndarray
    components: np.ndarray
    full_dofs: np.ndarray


@dataclass(frozen=True)
class CalculixMatrixStorage:
    K: sp.csr_matrix
    M: sp.csr_matrix
    dof_map: CalculixDofMap
    sti_path: Path
    mas_path: Path
    dof_path: Path


@dataclass(frozen=True)
class CalculixNodeHistory:
    times: np.ndarray
    displacements: dict[int, np.ndarray]


def write_cantilever_matrix_storage_input(
    path: str | Path,
    mesh: Mesh,
    fixed_nodes: np.ndarray,
    young_modulus: float = 2.1e5,
    poisson_ratio: float = 0.3,
    density: float = 7.8e-9,
) -> Path:
    """Write a CalculiX C3D8 input deck that emits K/M through MATRIXSTORAGE."""

    if mesh.element_type != "hex8":
        raise ValueError("CalculiX writer currently expects a hex8 mesh")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed_nodes = np.asarray(fixed_nodes, dtype=np.int64)

    lines: list[str] = [
        "*HEADING",
        "Modal contact ROM CalculiX matrix-storage cantilever",
        "*NODE, NSET=NALL",
    ]
    for idx, xyz in enumerate(mesh.nodes, start=1):
        lines.append(f"{idx}, {xyz[0]:.16g}, {xyz[1]:.16g}, {xyz[2]:.16g}")

    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    for elem_id, element in enumerate(mesh.elements, start=1):
        ccx_nodes = ", ".join(str(int(node_id) + 1) for node_id in element)
        lines.append(f"{elem_id}, {ccx_nodes}")

    lines.append("*NSET, NSET=FIXED")
    fixed_ccx = [str(int(node_id) + 1) for node_id in fixed_nodes]
    lines.extend(_wrap_csv(fixed_ccx))
    lines.extend(
        [
            "*MATERIAL, NAME=ELASTIC_MAT",
            "*ELASTIC",
            f"{young_modulus:.16g}, {poisson_ratio:.16g}",
            "*DENSITY",
            f"{density:.16g}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=ELASTIC_MAT",
            "*BOUNDARY",
            "FIXED, 1, 3",
            "*STEP",
            "*FREQUENCY, SOLVER=MATRIXSTORAGE",
            "1",
            "*END STEP",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_cantilever_dynamic_cload_input(
    path: str | Path,
    mesh: Mesh,
    fixed_nodes: np.ndarray,
    cload: np.ndarray,
    observed_nodes: np.ndarray,
    time_step: float,
    total_time: float,
    young_modulus: float = 2.1e5,
    poisson_ratio: float = 0.3,
    density: float = 7.8e-9,
    max_increments: int | None = None,
) -> Path:
    """Write a CalculiX direct-dynamic C3D8 deck with constant nodal loads."""

    if mesh.element_type != "hex8":
        raise ValueError("CalculiX writer currently expects a hex8 mesh")
    if time_step <= 0.0 or total_time <= 0.0:
        raise ValueError("time_step and total_time must be positive")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed_nodes = np.asarray(fixed_nodes, dtype=np.int64)
    observed_nodes = np.asarray(observed_nodes, dtype=np.int64)
    cload = np.asarray(cload, dtype=float)
    if cload.shape != (mesh.n_dofs,):
        raise ValueError(f"cload must have shape ({mesh.n_dofs},)")
    if max_increments is None:
        max_increments = int(np.ceil(total_time / time_step)) + 5

    lines: list[str] = [
        "*HEADING",
        "Modal contact ROM CalculiX direct dynamic benchmark",
        "*NODE, NSET=NALL",
    ]
    for idx, xyz in enumerate(mesh.nodes, start=1):
        lines.append(f"{idx}, {xyz[0]:.16g}, {xyz[1]:.16g}, {xyz[2]:.16g}")

    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    for elem_id, element in enumerate(mesh.elements, start=1):
        ccx_nodes = ", ".join(str(int(node_id) + 1) for node_id in element)
        lines.append(f"{elem_id}, {ccx_nodes}")

    lines.append("*NSET, NSET=FIXED")
    lines.extend(_wrap_csv([str(int(node_id) + 1) for node_id in fixed_nodes]))
    lines.append("*NSET, NSET=OBS")
    lines.extend(_wrap_csv([str(int(node_id) + 1) for node_id in observed_nodes]))
    lines.extend(
        [
            "*MATERIAL, NAME=ELASTIC_MAT",
            "*ELASTIC",
            f"{young_modulus:.16g}, {poisson_ratio:.16g}",
            "*DENSITY",
            f"{density:.16g}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=ELASTIC_MAT",
            "*BOUNDARY",
            "FIXED, 1, 3",
            f"*STEP, INC={max_increments}",
            "*DYNAMIC, DIRECT",
            f"{time_step:.16g}, {total_time:.16g}",
            "*CLOAD",
        ]
    )
    for dof, value in enumerate(cload):
        if abs(value) <= 1.0e-14:
            continue
        node_id = dof // 3
        component = dof % 3
        lines.append(f"{node_id + 1}, {component + 1}, {value:.16g}")
    lines.extend(["*NODE PRINT, NSET=OBS, FREQUENCY=1", "U", "*END STEP"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ccx_wsl(
    job_name: str,
    workdir: str | Path,
    timeout: int | None = 120,
    distribution: str | None = None,
    ccx_command: str = "ccx",
) -> CalculixRunResult:
    """Run CalculiX from WSL while keeping job files in the Windows workspace."""

    workdir = Path(workdir).resolve()
    stem = Path(job_name).stem
    wsl_dir = windows_to_wsl_path(workdir)
    shell_command = f"cd {shlex.quote(wsl_dir)} && {shlex.quote(ccx_command)} -i {shlex.quote(stem)}"
    command = ["wsl"]
    if distribution:
        command.extend(["-d", distribution])
    command.extend(["bash", "-lc", shell_command])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    result = CalculixRunResult(
        job_name=stem,
        workdir=workdir,
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CalculiX failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return result


def windows_to_wsl_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    windows_path = PureWindowsPath(str(resolved))
    drive = windows_path.drive.rstrip(":").lower()
    if drive:
        parts = [part for part in windows_path.parts[1:] if part not in {"\\", "/"}]
        return "/mnt/" + drive + "/" + "/".join(parts)

    completed = subprocess.run(
        ["wsl", "wslpath", "-a", str(resolved)],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def read_matrix_storage(job_name: str, workdir: str | Path) -> CalculixMatrixStorage:
    """Read CalculiX MATRIXSTORAGE files: job.sti, job.mas and job.dof."""

    workdir = Path(workdir)
    stem = Path(job_name).stem
    sti_path = workdir / f"{stem}.sti"
    mas_path = workdir / f"{stem}.mas"
    dof_path = workdir / f"{stem}.dof"
    dof_map = read_dof_map(dof_path)
    shape = (dof_map.full_dofs.size, dof_map.full_dofs.size)
    return CalculixMatrixStorage(
        K=read_ccx_coordinate_matrix(sti_path, shape=shape, symmetric=True),
        M=read_ccx_coordinate_matrix(mas_path, shape=shape, symmetric=True),
        dof_map=dof_map,
        sti_path=sti_path,
        mas_path=mas_path,
        dof_path=dof_path,
    )


def read_node_print_displacements(path: str | Path) -> CalculixNodeHistory:
    """Parse CalculiX .dat displacement blocks from *NODE PRINT output."""

    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header = re.compile(r"displacements .* time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    times: list[float] = []
    displacements: dict[int, list[np.ndarray]] = {}
    current_time: float | None = None
    for line in text:
        match = header.search(line)
        if match:
            current_time = float(match.group(1).replace("D", "E"))
            times.append(current_time)
            continue
        if current_time is None:
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            node_id = int(parts[0]) - 1
            values = np.asarray([float(part.replace("D", "E")) for part in parts[1:]], dtype=float)
        except ValueError:
            continue
        displacements.setdefault(node_id, []).append(values)

    time_array = np.asarray(times, dtype=float)
    return CalculixNodeHistory(
        times=time_array,
        displacements={node_id: np.asarray(values, dtype=float) for node_id, values in displacements.items()},
    )


def read_dof_map(path: str | Path) -> CalculixDofMap:
    node_ids: list[int] = []
    components: list[int] = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        token = raw_line.strip()
        if not token:
            continue
        node_id, component = _parse_dof_token(token)
        node_ids.append(node_id)
        components.append(component)

    node_array = np.asarray(node_ids, dtype=np.int64)
    component_array = np.asarray(components, dtype=np.int64)
    full_dofs = 3 * node_array + component_array
    return CalculixDofMap(node_array, component_array, full_dofs)


def read_ccx_coordinate_matrix(
    path: str | Path,
    shape: tuple[int, int] | None = None,
    symmetric: bool = False,
) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw_line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        row = int(parts[0]) - 1
        col = int(parts[1]) - 1
        value = float(parts[2].replace("D", "E").replace("d", "E"))
        rows.append(row)
        cols.append(col)
        data.append(value)
        if symmetric and row != col:
            rows.append(col)
            cols.append(row)
            data.append(value)

    if shape is None:
        size = max(max(rows, default=-1), max(cols, default=-1)) + 1
        shape = (size, size)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    matrix.eliminate_zeros()
    return matrix


def matrix_storage_to_fem_system(mesh: Mesh, matrices: CalculixMatrixStorage) -> FEMSystem:
    """Expand CalculiX active-DOF matrices to the package's full 3-DOF layout."""

    full_shape = (mesh.n_dofs, mesh.n_dofs)
    active = matrices.dof_map.full_dofs
    K = _expand_active_matrix(matrices.K, active, full_shape)
    M = _expand_active_matrix(matrices.M, active, full_shape)
    all_dofs = np.arange(mesh.n_dofs, dtype=np.int64)
    fixed_dofs = np.setdiff1d(all_dofs, active, assume_unique=False)
    fixed_nodes = np.unique(fixed_dofs // 3)
    return FEMSystem(mesh, K, M, fixed_nodes, fixed_dofs, active)


def _expand_active_matrix(matrix: sp.csr_matrix, active_dofs: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    coo = matrix.tocoo()
    rows = active_dofs[coo.row]
    cols = active_dofs[coo.col]
    return sp.coo_matrix((coo.data, (rows, cols)), shape=shape).tocsr()


def _parse_dof_token(token: str) -> tuple[int, int]:
    if "." not in token:
        raise ValueError(f"invalid CalculiX dof token: {token!r}")
    node_text, component_text = token.split(".", 1)
    return int(node_text) - 1, int(component_text[0]) - 1


def _wrap_csv(values: list[str], width: int = 12) -> list[str]:
    lines = []
    for start in range(0, len(values), width):
        lines.append(", ".join(values[start : start + width]))
    return lines
