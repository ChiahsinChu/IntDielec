import numpy as np

from ..exts.toolbox.toolbox.utils.math import gaussian_func


class WaterEDen:
    def __init__(self, atoms) -> None:
        self.atoms = atoms

    def run(self, grid, O_sigma=0.669, H_sigma=0.371, z_ref=0.0):
        """
        *_sigma: atomic radius [Å]
        """
        atype = np.array(self.atoms.get_chemical_symbols())
        O_mask = (atype == "O")
        H_mask = (atype == "H")
        O_coord = self.atoms.get_positions()[:, 2][O_mask] - z_ref
        H_coord = self.atoms.get_positions()[:, 2][H_mask] - z_ref
        out = np.zeros_like(grid)
        for z in O_coord:
            out += 6 * gaussian_func(grid, z, O_sigma)
        for z in H_coord:
            out += gaussian_func(grid, z, H_sigma)
        return out