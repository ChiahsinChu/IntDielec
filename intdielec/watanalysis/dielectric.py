import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.units import constants, convert
from scipy import integrate


class InverseDielectricConstant(AnalysisBase):
    def __init__(
        self,
        atomgroups,
        bin_edges,
        surf_ids,
        axis: int = 2,
        temperature=330,
        img_plane=0.,
        make_whole=False,
        verbose=False,
    ) -> None:
        self.universe = atomgroups.universe
        super().__init__(self.universe.trajectory, verbose)

        self.atoms = atomgroups
        self.bin_width = bin_edges[1] - bin_edges[0]
        self.bins = (bin_edges[1:] + bin_edges[:-1]) / 2
        self.nbins = len(bin_edges) - 1
        self.surf_ids = surf_ids
        self.axis = axis
        self.temperature = temperature
        self.img_plane = img_plane
        self.make_whole = make_whole

    def _prepare(self):
        if not hasattr(self.atoms, "charges"):
            raise NoDataError("No charges defined given atomgroup.")

        if not np.allclose(
                self.atoms.total_charge(compound='fragments'), 0.0, atol=1E-5):
            raise NotImplementedError("Analysis for non-neutral systems or"
                                      " systems with free charges are not"
                                      " available.")

        self.results.m = np.zeros((self.nbins))
        self.results.mM = np.zeros((self.nbins))
        self.results.M = 0.
        self.results.M2 = 0.
        self.results.volume = 0.

    def _single_frame(self):
        if self.make_whole:
            self.atoms.unwrap()

        ave_axis = np.delete(np.arange(3), self.axis)
        ts_area = self._ts.dimensions[ave_axis[0]] * self._ts.dimensions[
            ave_axis[1]]

        # get refs
        z = self._ts.positions[:, self.axis]
        _z_lo = np.mean(z[self.surf_ids[0]])
        _z_hi = np.mean(z[self.surf_ids[1]])
        z_lo = np.min([_z_lo, _z_hi])
        z_hi = np.max([_z_lo, _z_hi])
        # print(z_lo, z_hi)

        # # M
        # M = np.dot(self.universe.atoms.charges,
        #            self.universe.atoms.positions)[self.axis]

        bin_edges = np.linspace(z_lo, z_hi,
                                int((z_hi - z_lo) / self.bin_width) + 1)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2.

        # charge density [e/A^3]
        rho, bin_edges = np.histogram(self.atoms.positions[:, 2],
                                      bins=bin_edges,
                                      weights=self.atoms.charges)
        bin_volumes = np.diff(bin_edges) * ts_area
        rho /= bin_volumes
        _m = -integrate.cumulative_trapezoid(rho, bins, initial=0)

        # M
        M = np.sum(_m * bin_volumes)
        self.results.M += M
        self.results.M2 += (M**2)

        # lo surf
        m = np.interp(self.bins + z_lo, bins, _m)
        self.results.m += m
        self.results.mM += (m * M)
        # hi surf
        m = np.interp(np.sort(z_hi - self.bins), bins, _m)
        self.results.m += np.flip(m)
        self.results.mM += np.flip(m * M)

        ts_volume = ts_area * (z_hi - z_lo - 2 * self.img_plane)
        self.results.volume += ts_volume

    def _conclude(self):
        self.results.m /= (self.n_frames * 2)
        self.results.mM /= (self.n_frames * 2)
        self.results.M /= self.n_frames
        self.results.M2 /= self.n_frames
        self.results.volume /= self.n_frames

        x_fluct = self.results.mM - self.results.m * self.results.M
        M_fluct = self.results.M2 - self.results.M * self.results.M
        const = convert(
            constants["Boltzman_constant"], "kJ/mol",
            "eV") * self.temperature * constants["electric_constant"]
        self.results.inveps = 1 - x_fluct / (const +
                                             M_fluct / self.results.volume)

        self.results.bins = self.bins
        self.results.temperature = self.temperature


class AdInverseDielectricConstant(InverseDielectricConstant):
    def __init__(self,
                 atomgroups,
                 bin_edges,
                 surf_ids,
                 cutoff=2.7,
                 sfactor=2,
                 calc_unscaled=False,
                 **kwargs) -> None:
        super().__init__(atomgroups, bin_edges, surf_ids, **kwargs)
        self.cutoff = cutoff
        self.sfactor = sfactor
        self.calc_unscaled = calc_unscaled

    def _prepare(self):
        super()._prepare()
        self.results.m_scaled = np.zeros((self.nbins))
        self.results.mM_scaled = np.zeros((self.nbins))
        self.results.M_scaled = 0.
        self.results.M2_scaled = 0.

    def _single_frame(self):
        if self.make_whole:
            self.atoms.unwrap()

        ave_axis = np.delete(np.arange(3), self.axis)
        ts_area = self._ts.dimensions[ave_axis[0]] * self._ts.dimensions[
            ave_axis[1]]

        # get refs
        z = self._ts.positions[:, self.axis]
        _z_lo = np.mean(z[self.surf_ids[0]])
        _z_hi = np.mean(z[self.surf_ids[1]])
        z_lo = np.min([_z_lo, _z_hi])
        z_hi = np.max([_z_lo, _z_hi])
        # print(z_lo, z_hi)
        bin_edges = np.linspace(z_lo, z_hi,
                                int((z_hi - z_lo) / self.bin_width) + 1)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2.
        ts_volume = ts_area * (z_hi - z_lo - 2 * self.img_plane)
        self.results.volume += ts_volume

        if self.calc_unscaled:
            # unscaled charges
            # charge density [e/A^3]
            rho, bin_edges = np.histogram(self.atoms.positions[:, 2],
                                          bins=bin_edges,
                                          weights=self.atoms.charges)
            bin_volumes = np.diff(bin_edges) * ts_area
            rho /= bin_volumes
            _m = -integrate.cumulative_trapezoid(rho, bins, initial=0)
            # M
            M = np.sum(_m * bin_volumes)
            self.results.M += M
            self.results.M2 += (M**2)
            # lo surf m
            m = np.interp(self.bins + z_lo, bins, _m)
            self.results.m += m
            self.results.mM += (m * M)
            # hi surf m
            m = np.interp(np.sort(z_hi - self.bins), bins, _m)
            self.results.m += np.flip(m)
            self.results.mM += np.flip(m * M)

        # scaled charges
        # scale charges of chemisorbed water
        atype_mask = (self.universe.atoms.types == "O")
        z_mask = ((z <= (z_lo + self.cutoff)) | (z >= (z_hi - self.cutoff)))
        sel_O_ids = np.nonzero(atype_mask * z_mask)[0]
        sel_ids = np.sort(
            np.concatenate([sel_O_ids, sel_O_ids + 1, sel_O_ids + 2]))
        charges = self.atoms.charges.copy()
        charges[sel_ids] *= self.sfactor

        # charge density [e/A^3]
        rho, bin_edges = np.histogram(self.atoms.positions[:, 2],
                                      bins=bin_edges,
                                      weights=charges)
        bin_volumes = np.diff(bin_edges) * ts_area
        rho /= bin_volumes
        _m = -integrate.cumulative_trapezoid(rho, bins, initial=0)
        # M
        M = np.sum(_m * bin_volumes)
        self.results.M_scaled += M
        self.results.M2_scaled += (M**2)
        # lo surf m
        m = np.interp(self.bins + z_lo, bins, _m)
        self.results.m_scaled += m
        self.results.mM_scaled += (m * M)
        # hi surf m
        m = np.interp(np.sort(z_hi - self.bins), bins, _m)
        self.results.m_scaled += np.flip(m)
        self.results.mM_scaled += np.flip(m * M)

    def _conclude(self):
        super()._conclude()

        self.results.m_scaled /= (self.n_frames * 2)
        self.results.mM_scaled /= (self.n_frames * 2)
        self.results.M_scaled /= self.n_frames
        self.results.M2_scaled /= self.n_frames

        x_fluct = self.results.mM_scaled - self.results.m_scaled * self.results.M_scaled
        M_fluct = self.results.M2_scaled - self.results.M_scaled * self.results.M_scaled
        const = convert(
            constants["Boltzman_constant"], "kJ/mol",
            "eV") * self.temperature * constants["electric_constant"]
        self.results.inveps_scaled = 1 - x_fluct / (
            const + M_fluct / self.results.volume)
