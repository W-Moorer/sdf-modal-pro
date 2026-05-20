"""Modal contact ROM first-version prototype."""

from modal_contact_rom.fem_io.generate import FEMSystem, cantilever_block
from modal_contact_rom.fem_io.mesh import Mesh

__all__ = ["FEMSystem", "Mesh", "cantilever_block"]

