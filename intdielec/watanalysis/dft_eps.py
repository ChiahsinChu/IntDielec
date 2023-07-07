from tqdm import trange

from ..exts.toolbox.toolbox.io.cp2k import Cp2kCube
from ..exts.toolbox.toolbox.utils import *
from ..exts.toolbox.toolbox.utils.unit import *
from ..exts.toolbox.toolbox.utils.utils import save_dict


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
    

class DFTInvEps:
    def __init__(self, fnames, **kwargs) -> None:
        self.fnames = fnames
        self.fmt = os.path.splitext(fnames[0])[1][1:]
        self.kwargs = kwargs

        output = self.read_data(fnames[0])
        self.m, self.M = self.calc_local_m(output)
        self.mM = self.m * self.M
        self.M2 = self.M ** 2

        self.const = (constants.epsilon_0 / constants.elementary_charge *
                      constants.angstrom) * constants.physical_constants[
                          "Boltzmann constant in eV/K"][0]
    
    def run(self):
        nframes = len(self.fnames)
        for ii in trange(1, nframes):
            output = self.read_data(self.fnames[ii])
            m_perp, M_perp = self.calc_local_m(output)
            self.m += m_perp
            self.M += M_perp
            self.mM += m_perp * M_perp
            self.M2 += M_perp ** 2

        self.m /= nframes
        self.M /= nframes
        self.mM /= nframes
        self.M2 /= nframes

        bins = output[0]
        self.results = {
            "bins": bins,
            "m": self.m,
            "M": self.M,
            "mM": self.mM,
            "M2": self.M2
        }
    
    def save(self, fname):
        np.save(fname, self.results)

    def calc_inveps(self, volume, temperature=330, fname=None):
        try:
            results = self.results
        except:
            results = load_dict(fname)
        
        m = results["m"]
        M = results["M"]
        mM = results["mM"]
        M2 = results["M2"]
        x_fluct = mM - m * M
        M_fluct = M_fluct = M2 - M ** 2
        inveps = 1 -  x_fluct / (self.const * temperature + M_fluct / volume)
        return inveps

    @staticmethod
    def calc_local_m(output):
        m_perp = np.cumsum(output[1])
        m_perp *= (output[0][1] - output[0][0])
        M_perp = np.sum(m_perp)
        return m_perp, M_perp
    
    def read_data(self, fname):
        try:
            return getattr(self, "read_data_%s" % self.fmt)(fname)
        except:
            raise AttributeError("Unknown format %s" % self.fmt)

    def read_data_cube(self, fname):
        cube = Cp2kCube(fname)
        output = cube.get_ave_cube(**self.kwargs)
        np.save(os.path.join(os.path.dirname(fname), "totden.npy"), [output[0], output[1]])
        return [output[0], output[1]]
    
    def read_data_npy(self, fname):
        output = np.load(fname)
        return output