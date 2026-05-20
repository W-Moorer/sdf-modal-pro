from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.modal_ilc import build_normal_patch_load_bases, write_patch_load_basis_tables
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface


def main() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    medium = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2)).level("medium")
    load_bases = build_normal_patch_load_bases(fem.mesh, surface, medium.patches)
    output_dir = Path("outputs/load_basis")
    write_patch_load_basis_tables(load_bases, medium.patches, output_dir)

    print(f"wrote {output_dir / 'load_basis_stats.csv'}")
    print(f"wrote {output_dir / 'basis_condition.csv'}")
    print(f"wrote {output_dir / 'load_basis_summary.md'}")


if __name__ == "__main__":
    main()
