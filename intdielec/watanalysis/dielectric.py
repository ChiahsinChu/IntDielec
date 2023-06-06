from ase import geometry
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.exceptions import NoDataError
from scipy import constants, integrate

from ..exts.toolbox.toolbox.utils import *
from ..exts.toolbox.toolbox.utils.utils import save_dict

try:
    from deepmd.infer import DeepDipole
except:
    pass


class InverseDielectricConstant(AnalysisBase):
    def __init__(
        self,
        atomgroups,
        bin_width,
        surf_ids,
        axis: int = 2,
        temperature=330,
        img_plane=0.,
        make_whole=False,
        dimensions=None,
        verbose=True,
    ) -> None:
        self.universe = atomgroups.universe
        super().__init__(self.universe.trajectory, verbose)

        try:
            l_box = self.universe.trajectory.dimensions[axis]
        except:
            l_box = dimensions[axis]

        self.nbins = int(l_box / bin_width)
        self.bin_edges = np.linspace(0, l_box, self.nbins + 1)
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
        self.bins = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.

        self.atoms = atomgroups
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

        if not np.allclose(self.atoms.total_charge(), 0.0, atol=1E-5):
            raise NotImplementedError("Analysis for non-neutral systems or"
                                      " systems with free charges are not"
                                      " available.")

        self.results.m = np.zeros((self.nbins))
        self.results.mM = np.zeros((self.nbins))
        self.results.M = 0.
        self.results.M2 = 0.

        self.results.z_lo = 0.
        self.results.z_hi = 0.

    def _single_frame(self):
        if self.make_whole:
            self.atoms.unwrap()

        # get refs
        z = self._ts.positions[:, self.axis]
        _z_lo = np.mean(z[self.surf_ids[0]])
        _z_hi = np.mean(z[self.surf_ids[1]])
        self._ts_z_lo = np.min([_z_lo, _z_hi])
        self._ts_z_hi = np.max([_z_lo, _z_hi])
        self.results.z_lo += self._ts_z_lo
        self.results.z_hi += self._ts_z_hi

        # charge density [e/A^3]
        rho = self._calc_rho()
        bin_volumes = np.diff(self.bin_edges) * self.cross_area
        rho /= bin_volumes
        m = -integrate.cumulative_trapezoid(rho, self.bins, initial=0)
        self.results.m += m

        # M
        # M = np.dot(self.universe.atoms.charges,
        #            self.universe.atoms.positions)[self.axis]
        M = np.sum(m * bin_volumes)
        self.results.M += M
        self.results.M2 += (M**2)
        self.results.mM += (m * M)

    def _conclude(self):
        self.results.m /= self.n_frames
        self.results.mM /= self.n_frames
        self.results.M /= self.n_frames
        self.results.M2 /= self.n_frames

        self.results.z_lo /= self.n_frames
        self.results.z_hi /= self.n_frames
        self.results.volume = self.cross_area * (
            self.results.z_hi - self.results.z_lo - 2 * self.img_plane)

        x_fluct = self.results.mM - self.results.m * self.results.M
        M_fluct = self.results.M2 - self.results.M * self.results.M
        self.results.inveps = 1 - x_fluct / (self.const +
                                             M_fluct / self.results.volume)

        self.results.bins = self.bins
        self.results.temperature = self.temperature

    def save(self, save_dir=".", prefix=None):
        if prefix is not None:
            fname = "%s_inveps.pkl" % prefix
        else:
            fname = "inveps.pkl"
        save_dict(self.results, os.path.join(save_dir, fname))

    @property
    def cross_area(self):
        ave_axis = np.delete(np.arange(3), self.axis)
        ds = self._ts.dimensions
        cross_area = ds[ave_axis[0]] * ds[ave_axis[1]]
        return cross_area

    def _calc_rho(self):
        """
        rho: charge density [e/A^3]
        """
        rho, bin_edges = np.histogram(self.atoms.positions[:, self.axis],
                                      bins=self.bin_edges,
                                      weights=self.atoms.charges)
        return rho


class AdInverseDielectricConstant(InverseDielectricConstant):
    def __init__(self,
                 atomgroups,
                 bin_width,
                 surf_ids,
                 cutoff=2.7,
                 sfactor=2,
                 perturbation=0,
                 calc_unscaled=False,
                 **kwargs) -> None:
        super().__init__(atomgroups, bin_width, surf_ids, **kwargs)
        self.cutoff = cutoff
        self.sfactor = sfactor
        self.perturbation = perturbation
        self.calc_unscaled = calc_unscaled

    def _prepare(self):
        super()._prepare()
        self.results.m_unscaled = np.zeros((self.nbins))
        self.results.mM_unscaled = np.zeros((self.nbins))
        self.results.M_unscaled = 0.
        self.results.M2_unscaled = 0.

    def _single_frame(self):
        super()._single_frame()

        if self.calc_unscaled:
            # charge density [e/A^3]
            rho, bin_edges = np.histogram(self.atoms.positions[:, 2],
                                          bins=bin_edges,
                                          weights=self.atoms.charges)
            bin_volumes = np.diff(bin_edges) * self.cross_area
            rho /= bin_volumes
            m = -integrate.cumulative_trapezoid(rho, self.bins, initial=0)
            self.results.m_unscaled += m

            # M
            M = np.sum(m * bin_volumes)
            self.results.M_unscaled += M
            self.results.M2_unscaled += (M**2)
            self.results.mM_unscaled += (m * M)

    def _conclude(self):
        super()._conclude()
        if self.calc_unscaled:
            self.results.m_unscaled /= self.n_frames
            self.results.mM_unscaled /= self.n_frames
            self.results.M_unscaled /= self.n_frames
            self.results.M2_unscaled /= self.n_frames

            x_fluct = self.results.mM_unscaled - self.results.m_unscaled * self.results.M_unscaled
            M_fluct = self.results.M2_unscaled - self.results.M_unscaled * self.results.M_unscaled
            self.results.inveps_unscaled = 1 - x_fluct / (
                self.const + M_fluct / self.results.volume)

    def _calc_rho(self):
        z = self.atoms.positions[:, self.axis]
        # scale charges of chemisorbed water
        atype_mask = (self.atoms.types == "O")
        z_mask = ((z <= (self._ts_z_lo + self.cutoff)) |
                  (z >= (self._ts_z_hi - self.cutoff)))
        sel_O_ids = np.nonzero(atype_mask * z_mask)[0]
        sel_ids = np.sort(
            np.concatenate([sel_O_ids, sel_O_ids + 1, sel_O_ids + 2]))
        charges = self.atoms.charges.copy()
        charges[sel_ids] *= self.sfactor
        # add pert in charge and keep electroneutrality
        perturbation = np.random.uniform(low=-self.perturbation,
                                         high=self.perturbation,
                                         size=len(sel_ids))
        perturbation -= perturbation.mean()
        charges[sel_ids] += perturbation

        # charge density [e/A^3]
        rho, bin_edges = np.histogram(z, bins=self.bin_edges, weights=charges)
        return rho


class DPInverseDielectricConstant(InverseDielectricConstant):
    def __init__(self, atomgroups, bin_width, surf_ids, model,
                 **kwargs) -> None:
        super().__init__(atomgroups, bin_width, surf_ids, **kwargs)
        self.model = DeepDipole(model)
        atype = atomgroups.types
        self.atype = np.ones(len(atomgroups), dtype=np.int32)
        self.atype[atype == "O"] = 0
        self.atype[atype == "H"] = 1

    def _calc_rho(self):
        """
        rho: charge density [e/A^3]
        """
        coord = self.atoms.positions

        charges = np.ones(len(self.atoms))
        # print(coord.shape, charges.shape)

        # O
        charges[self.atype == 0] = 6
        n_rho, bin_edges = np.histogram(coord[:, self.axis],
                                        bins=self.bin_edges,
                                        weights=charges)

        wannier = self._dp_eval(coord.reshape(1, -1))
        e_coord = coord[self.atype == 0] + wannier
        e_charges = np.full((len(wannier)), -8)
        e_rho, bin_edges = np.histogram(e_coord[:, self.axis],
                                        bins=self.bin_edges,
                                        weights=e_charges)

        return n_rho + e_rho

    def _dp_eval(self, coord):
        cell = geometry.cell.cellpar_to_cell(self._ts.dimensions)
        dipole = self.model.eval(coord, cell.reshape(1, 9), self.atype)
        return dipole.reshape(-1, 3)


class DeprecatedInverseDielectricConstant(AnalysisBase):
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

        if not np.allclose(self.atoms.total_charge(), 0.0, atol=1E-5):
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


class DeprecatedAdInverseDielectricConstant(DeprecatedInverseDielectricConstant
                                            ):
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
