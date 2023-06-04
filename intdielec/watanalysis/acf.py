"""
Ref:
- https://docs.mdanalysis.org/2.3.0/documentation_pages/analysis/waterdynamics.html
"""
from collections import deque

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core import AtomGroup

from ..exts.toolbox.toolbox.utils import *
from ..exts.toolbox.toolbox.utils.math import handle_zero_division


class SelectedTimeSeries(AnalysisBase):
    def __init__(self, atomgroup: AtomGroup, dts=None, verbose=True, **kwargs):
        self.universe = atomgroup.universe
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)
        self.ag = atomgroup
        self.dts = np.sort(dts)
        self.n_dt = len(self.dts)
        # n_sample can be modified in the child class
        self.n_sample = len(self.ag)

    def _prepare(self):
        # save raw data
        self.raw_data = []
        maxlen = int(self.dts[-1] + 1)
        for ii in range(self.n_sample):
            self.raw_data.append(deque(maxlen=maxlen))

        self.score = np.zeros((self.n_dt, self.n_sample))
        self.count = np.zeros(self.n_dt)
        self.acf = np.zeros(self.n_dt)

    def _single_frame(self):
        # calculate interested quantity for all sample
        ts_data = self._calc_ts_data()
        assert len(ts_data) == self.n_sample

        # update self.raw_data
        for queue, data in zip(self.raw_data, ts_data):
            queue.append(data)

        # make mask for the *selected* sample in the current frame
        ts_mask = self._calc_ts_mask()
        sel_ids = np.nonzero(ts_mask)[0]
        rest_ids = np.nonzero(~ts_mask)[0]
        for ii in range(self.n_dt):
            _score = self.score[ii]
            dt = self.dts[ii]

            # update score for every dt
            _score[sel_ids] += 1
            _score[rest_ids] = -dt
            # print("after:\n", _score)
            # get mask from score
            output_mask = (_score >= 0)
            # update self.count and self.acf
            self._update_output(ii, output_mask)

    def _conclude(self):
        self.acf = handle_zero_division(self.acf, self.count)

    def save(self, fname="acf.txt"):
        np.savetxt(fname, np.transpose([self.dts, self.acf]), header="dt acf")

    def _calc_ts_data(self):
        pass

    def _calc_ts_mask(self):
        pass

    def _update_output(self, ii, mask):
        pass

    def _calc_refs(self):
        refs = []
        for ref_ag in self.refs:
            refs.append(ref_ag.centroid()[self.axis])
        return refs


class SelectedDipoleACF(SelectedTimeSeries):
    def __init__(self,
                 atomgroup: AtomGroup,
                 dts=None,
                 refs=None,
                 cutoff=None,
                 axis=2,
                 verbose=True,
                 **kwargs):
        super().__init__(atomgroup, dts, verbose, **kwargs)
        # update n_sample
        self.n_sample = int(len(self.ag) / 3)

        self.refs = refs
        self.cutoff = cutoff
        self.axis = axis

    def _calc_ts_data(self):
        """
        only for unwrapped, OHH 
        """
        ts_positions = self.ag.positions
        ts_p_O = ts_positions[::3]
        ts_p_H1 = ts_positions[1::3]
        ts_p_H2 = ts_positions[2::3]
        dipole = ts_p_H1 + ts_p_H2 - 2 * ts_p_O
        dipole = np.reshape(dipole, (self.n_sample, 3))
        dipole = dipole / np.linalg.norm(dipole, axis=-1, keepdims=True)
        return dipole

    def _calc_ts_mask(self):
        if self.refs is not None:
            refs = self._calc_refs()
            ts_positions = self.ag.positions[::3, self.axis]
            ts_mask = np.full(self.n_sample, False)
            for ref in refs:
                # n_ref * n_sample
                _mask = (np.abs(ts_positions - ref) <= self.cutoff)
                # n_sample
                ts_mask += _mask
            return ts_mask
        else:
            return np.full(self.n_sample, True)

    def _update_output(self, ii, mask):
        # -1 & -1-dt
        acf = 0
        dt = self.dts[ii]
        try:
            for jj in np.nonzero(mask)[0]:
                single_raw_data = self.raw_data[jj]
                acf += self.lg2(
                    np.dot(single_raw_data[-1], single_raw_data[-1 - dt]))
            self.count[ii] += np.count_nonzero(mask)
            self.acf[ii] += acf
        except:
            pass

    @staticmethod
    def lg2(x):
        """
        Second Legendre polynomial
        """
        return (3 * x * x - 1) / 2


class SelectedMSD(SelectedTimeSeries):
    def __init__(self,
                 atomgroup: AtomGroup,
                 dts=None,
                 refs=None,
                 cutoff=None,
                 axis=2,
                 verbose=True,
                 **kwargs):
        super().__init__(atomgroup, dts, verbose, **kwargs)

        self.refs = refs
        self.cutoff = cutoff
        self.axis = axis

    def _calc_ts_data(self):
        """
        only for unwrapped, OHH 
        """
        ts_positions = self.ag.positions[:, self.axis]
        return ts_positions

    def _calc_ts_mask(self):
        if self.refs is not None:
            refs = self._calc_refs()
            ts_positions = self.ag.positions[:, self.axis]
            ts_mask = np.full(self.n_sample, False)
            for ref in refs:
                # n_ref * n_sample
                _mask = (np.abs(ts_positions - ref) <= self.cutoff)
                # n_sample
                ts_mask += _mask
            return ts_mask
        else:
            return np.full(self.n_sample, True)

    def _update_output(self, ii, mask):
        # -1 & -1-dt
        acf = 0
        dt = self.dts[ii]
        try:
            for jj in np.nonzero(mask)[0]:
                single_raw_data = self.raw_data[jj]
                acf += np.square(single_raw_data[-1] -
                                 single_raw_data[-1 - dt])
            self.count[ii] += np.count_nonzero(mask)
            self.acf[ii] += acf
        except:
            pass

    def save(self, fname="msd.txt"):
        np.savetxt(fname, np.transpose([self.dts, self.acf]), header="dt msd")
