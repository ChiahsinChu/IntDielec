from collections import deque
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core import AtomGroup
from scipy import constants, integrate

from ..exts.toolbox.toolbox.utils import *
from ..exts.toolbox.toolbox.utils.utils import save_dict


class SelectedACF(AnalysisBase):
    def __init__(self, 
                 atomgroup:AtomGroup, 
                 refs,
                 cutoff, 
                 axis=2, 
                 dts=None, 
                 **kwargs):
        self.universe = atomgroup.universe
        super().__init__(self.universe.trajectory, **kwargs)
        self.ag = atomgroup
        self.refs = refs
        self.cutoff = cutoff
        self.axis = axis
        self.dts = np.sort(dts)
        self.n_wat = int(len(self.ag) / 3)

    def _prepare(self):
        # save raw data
        self.raw_data = []
        for ii in range(len(self.n_wat)):
            self.raw_data.append(deque(maxlen=self.dts[-1]))

        self.score = np.zeros((len(self.dts), self.n_wat))
        self.count = np.zeros(len(self.dts))
        self.acf = np.zeros(len(self.dts))

    def _single_frame(self):
        # calculate dipole vector for all water
        dipole_vecs = self._calc_dipole_vec()
        # update self.raw_data
        
        ts_z_O = self.ag.positions[::3][self.axis]
        refs = self._calc_refs()
        ts_mask = np.full(self.n_wat, True)
        for ref in refs:
            ts_mask *= (np.abs(ts_z_O - ref) >= self.cutoff)

        # update score
        # get mask from score
        # update self.count and self.acf

        pass
    
    def _conclude(self):
        return super()._conclude()


    def _calc_dipole_vec(self):
        """
        only for unwrapped, OHH 
        """
        ts_positions = self.ag.positions
        ts_p_O = ts_positions[::3]
        ts_p_H1 = ts_positions[1::3]
        ts_p_H2 = ts_positions[2::3]
        dipole = ts_p_H1 + ts_p_H2 - 2 * ts_p_O 
        return dipole
    
    def _calc_refs(self):
        refs = []
        for ref_ag in self.refs:
            refs.append(ref_ag.centroid()[self.axis])
        return refs
        