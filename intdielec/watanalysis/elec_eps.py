import numpy as np


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
            out += 6 * self.gaussian_dist(grid, z, O_sigma)
        for z in H_coord:
            out += self.gaussian_dist(grid, z, H_sigma)
        return out

    @staticmethod
    def gaussian_dist(x, mu, sigma):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))