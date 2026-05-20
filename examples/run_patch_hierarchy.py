from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_contact_rom.fem_io import cantilever_block
from modal_contact_rom.surface_patch import build_patch_hierarchy, extract_surface, write_patch_hierarchy_tables


def main() -> None:
    fem = cantilever_block(nx=4, ny=2, nz=2)
    surface = extract_surface(fem.mesh)
    hierarchy = build_patch_hierarchy(surface, coarse_bins=(1, 1), medium_bins=(2, 2))
    output_dir = Path("outputs/patch_hierarchy")
    write_patch_hierarchy_tables(hierarchy, output_dir)

    print(f"wrote {output_dir / 'patch_stats.csv'}")
    print(f"wrote {output_dir / 'patch_coverage.csv'}")
    print(f"wrote {output_dir / 'patch_summary.md'}")


if __name__ == "__main__":
    main()
