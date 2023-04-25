"""
Ref:
    https://maicos-devel.gitlab.io/maicos/explanations/dielectric.html
"""
import numpy as np
from maicos import DielectricPlanar
from maicos.core import PlanarBase
from maicos.lib.math import symmetrize
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.exceptions import NoDataError
from scipy import constants, integrate

from ..utils.mda import make_index_selection
from ..exts.toolbox.toolbox.utils.utils import save_dict


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

        self.const = (constants.epsilon_0 / constants.elementary_charge *
                      constants.angstrom) * constants.physical_constants[
                          "Boltzmann constant in eV/K"][0] * self.temperature

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
        self.results.inveps = 1 - x_fluct / (self.const +
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
        self.results.inveps_scaled = 1 - x_fluct / (
            self.const + M_fluct / self.results.volume)


class TestInverseDielectricConstant(DielectricPlanar):
    def __init__(self, atomgroups, surf_ids, img_plane=0., **kwargs) -> None:
        super().__init__(atomgroups, **kwargs)
        self.img_plane = img_plane

        self.surf_lo = atomgroups.universe.select_atoms(
            make_index_selection(surf_ids[0]))
        self.surf_hi = atomgroups.universe.select_atoms(
            make_index_selection(surf_ids[1]))

    def _prepare(self):
        super()._prepare()
        self.volume = 0.
        self.z_lo = []
        self.z_hi = []

    def _single_frame(self):
        super()._single_frame()

        z_lo = self.surf_lo.positions[:, self.dim].mean()
        z_hi = self.surf_hi.positions[:, self.dim].mean()
        # print(z_lo, z_hi)
        self.z_lo.append(z_lo)
        self.z_hi.append(z_hi)

    def _conclude(self):
        z_lo = np.mean(self.z_lo)
        z_hi = np.mean(self.z_hi)
        volume = self._obs.bin_area * (np.abs(z_hi - z_lo) -
                                       2 * self.img_plane)
        self.new_results = {
            "z_lo": z_lo,
            "z_lo_std": np.std(self.z_lo),
            "z_hi": z_hi,
            "z_hi_std": np.std(self.z_hi),
            "volume": volume
        }

        PlanarBase._conclude(self)

        pref = 1 / constants.epsilon_0
        pref /= constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= constants.angstrom / \
            (constants.elementary_charge)**2

        self.results.pref = pref
        # NOTE: modify volume!!!
        self.results.V = volume

        # Perpendicular component
        # =======================
        cov_perp = self.means.mM_perp \
            - self.means.m_perp \
            * self.means.M_perp

        # Using propagation of uncertainties
        dcov_perp = np.sqrt(self.sems.mM_perp**2 +
                            (self.means.M_perp * self.sems.m_perp)**2 +
                            (self.means.m_perp * self.sems.M_perp)**2)

        var_perp = self.means.M_perp_2 - self.means.M_perp**2

        cov_perp_self = self.means.mm_perp \
            - (self.means.m_perp**2 * self.means.bin_volume[0])
        cov_perp_coll = self.means.cmM_perp \
            - self.means.m_perp * self.means.cM_perp

        if not self.is_3d:
            self.results.eps_perp = -pref * cov_perp
            self.results.eps_perp_self = -pref * cov_perp_self
            self.results.eps_perp_coll = -pref * cov_perp_coll
            self.results.deps_perp = pref * dcov_perp
            if (self.vac):
                self.results.eps_perp *= 2. / 3.
                self.results.eps_perp_self *= 2. / 3.
                self.results.eps_perp_coll *= 2. / 3.
                self.results.deps_perp *= 2. / 3.

        else:
            self.new_results["cov_perp"] = cov_perp
            self.new_results["pref"] = pref
            self.new_results["var_perp"] = var_perp

            self.results.eps_perp = \
                - cov_perp / (pref**-1 + var_perp / self.results.V)
            self.results.deps_perp = pref * dcov_perp

            self.results.eps_perp_self = \
                (- pref * cov_perp_self) \
                / (1 + pref / self.results.V * var_perp)
            self.results.eps_perp_coll = \
                (- pref * cov_perp_coll) \
                / (1 + pref / self.results.V * var_perp)

        # Parallel component
        # ==================
        cov_par = np.zeros((self.n_bins, self.n_atomgroups))
        dcov_par = np.zeros((self.n_bins, self.n_atomgroups))
        cov_par_self = np.zeros((self.n_bins, self.n_atomgroups))
        cov_par_coll = np.zeros((self.n_bins, self.n_atomgroups))

        for i in range(self.n_atomgroups):
            cov_par[:, i] = 0.5 * (self.means.mM_par[:, i] - np.dot(
                self.means.m_par[:, :, i], self.means.M_par))

            # Using propagation of uncertainties
            dcov_par[:, i] = 0.5 * np.sqrt(
                self.sems.mM_par[:, i]**2 +
                np.dot(self.sems.m_par[:, :, i]**2, self.means.M_par**2) +
                np.dot(self.means.m_par[:, :, i]**2, self.sems.M_par**2))

            cov_par_self[:, i] = 0.5 * (self.means.mm_par[:, i] - np.dot(
                self.means.m_par[:, :, i], self.means.m_par[:, :,
                                                            i].sum(axis=0)))
            cov_par_coll[:, i] = \
                0.5 * (self.means.cmM_par[:, i]
                       - (self.means.m_par[:, :, i]
                       * self.means.cM_par[:, :, i]).sum(axis=1))

        self.results.eps_par = pref * cov_par
        self.results.deps_par = pref * dcov_par
        self.results.eps_par_self = pref * cov_par_self
        self.results.eps_par_coll = pref * cov_par_coll

        if self.sym:
            symmetrize(self.results.eps_perp, axis=0, inplace=True)
            symmetrize(self.results.deps_perp, axis=0, inplace=True)
            symmetrize(self.results.eps_perp_self, axis=0, inplace=True)
            symmetrize(self.results.eps_perp_coll, axis=0, inplace=True)

            symmetrize(self.results.eps_par, axis=0, inplace=True)
            symmetrize(self.results.deps_par, axis=0, inplace=True)
            symmetrize(self.results.eps_par_self, axis=0, inplace=True)
            symmetrize(self.results.eps_par_coll, axis=0, inplace=True)

        # Print Alex Schlaich citation
        # logger.info(citation_reminder("10.1103/PhysRevLett.117.048001"))

    def save(self):
        super().save()
        fname = "{}{}".format(self.output_prefix, "_raw_data.pkl")
        save_dict(self.new_results, fname)