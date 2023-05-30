import numpy as np
from maicos import DielectricPlanar
from maicos.core import PlanarBase
from maicos.lib.math import symmetrize
from scipy import constants

from ..exts.toolbox.toolbox.utils.utils import save_dict
from ..utils.mda import make_index_selection


class TestInverseDielectricConstant(DielectricPlanar):
    """
    Ref:
        https://maicos-devel.gitlab.io/maicos/explanations/dielectric.html
    """
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
