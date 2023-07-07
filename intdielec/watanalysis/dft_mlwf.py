from ..exts.toolbox.toolbox.utils import *


class WaterMLWF:
    def __init__(self, atoms) -> None:
        atoms.wrap()
        self.atoms = atoms
        
    def run(self):
        atype = np.array(self.atoms.get_chemical_symbols())
        O_ids = np.nonzero(atype == "O")[0]
        H_ids = np.nonzero(atype == "H")[0]
        e_ids = np.nonzero(atype == "X")[0]

        z = []
        dipole = []
        coord = self.atoms.get_positions()[:, 2]
        for ii in O_ids:
            H_coords = self.get_H_coords(self.atoms, ii, H_ids)
            e_coord = self.get_e_coord(self.atoms, ii, e_ids)
            _dipole = 2 * H_coords.mean(axis=0) - 8 * e_coord
            dipole.append(_dipole)
            z.append(coord[ii])

        self.results = {
            "dipole": np.array(dipole),
            "z": np.array(z)
        }
                
    @staticmethod
    def get_H_coords(atoms, O_id, H_ids):
        out = atoms.get_distances(O_id, H_ids, mic=True, vector=True)
        ds = np.linalg.norm(out, axis=-1)
        sel_ids = np.argsort(ds)[:2]
        return out[sel_ids]

    @staticmethod
    def get_e_coord(atoms, O_id, e_ids):
        out = atoms.get_distances(O_id, e_ids, mic=True, vector=True)
        ds = np.linalg.norm(out, axis=-1)
        sel_ids = np.argsort(ds)[:4]
        return out[sel_ids].mean(axis=0)