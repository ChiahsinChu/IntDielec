from ase import geometry
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.dielectric import DielectricConstant
from MDAnalysis.exceptions import NoDataError
from scipy import constants, integrate

from ..exts.toolbox.toolbox.utils import *
from ..exts.toolbox.toolbox.utils.math import gaussian_int
from ..exts.toolbox.toolbox.utils.utils import save_dict


class InverseDielectricConstant(AnalysisBase):
    def __init__(
        self,
        atomgroups,
        bin_width,
        surf_ids,
        axis: int = 2,
        temperature=330.0,
        img_plane=0.0,
        make_whole=False,
        dimensions=None,
        verbose=True,
    ) -> None:
        self.universe = atomgroups.universe
        super().__init__(self.universe.trajectory, verbose)

        try:
            l_box = self.universe.dimensions[axis]
        except:
            l_box = dimensions[axis]

        self.nbins = int(l_box / bin_width)
        self.bin_edges = np.linspace(0, l_box, self.nbins + 1)
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
        self.bins = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        self.atoms = atomgroups
        self.surf_ids = surf_ids
        self.axis = axis
        self.temperature = temperature
        self.img_plane = img_plane
        self.make_whole = make_whole

        self.const = (
            (constants.epsilon_0 / constants.elementary_charge * constants.angstrom)
            * constants.physical_constants["Boltzmann constant in eV/K"][0]
            * self.temperature
        )

    def _prepare(self):
        if not hasattr(self.atoms, "charges"):
            raise NoDataError("No charges defined given atomgroup.")

        if not np.allclose(self.atoms.total_charge(), 0.0, atol=1e-5):
            raise NotImplementedError(
                "Analysis for non-neutral systems or"
                " systems with free charges are not"
                " available."
            )

        self.results.m = np.zeros((self.nbins))
        self.results.mM = np.zeros((self.nbins))
        self.results.M = 0.0
        self.results.M2 = 0.0

        self.results.z_lo = 0.0
        self.results.z_hi = 0.0

        self.results.rho = np.zeros((self.nbins))

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
        self.results.rho += rho

        # M
        # M = np.dot(self.universe.atoms.charges,
        #            self.universe.atoms.positions)[self.axis]
        M = np.sum(m * bin_volumes)
        self.results.M += M
        self.results.M2 += M**2
        self.results.mM += m * M

    def _conclude(self):
        self.results.m /= self.n_frames
        self.results.mM /= self.n_frames
        self.results.M /= self.n_frames
        self.results.M2 /= self.n_frames
        self.results.rho /= self.n_frames

        self.results.z_lo /= self.n_frames
        self.results.z_hi /= self.n_frames
        self.results.volume = self.cross_area * (
            self.results.z_hi - self.results.z_lo - 2 * self.img_plane
        )

        x_fluct = self.results.mM - self.results.m * self.results.M
        M_fluct = self.results.M2 - self.results.M * self.results.M
        self.results.inveps = 1 - x_fluct / (self.const + M_fluct / self.results.volume)

        self.results.bins = self.bins
        self.results.temperature = self.temperature

    def save(self, save_dir=".", prefix=None):
        if prefix is not None:
            fname = "%s.inveps.pkl" % prefix
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
        rho, bin_edges = np.histogram(
            self.atoms.positions[:, self.axis],
            bins=self.bin_edges,
            weights=self.atoms.charges,
        )
        return rho


class AdInverseDielectricConstant(InverseDielectricConstant):
    def __init__(
        self,
        atomgroups,
        bin_width,
        surf_ids,
        cutoff=2.7,
        sfactor=2,
        perturbation=0,
        calc_unscaled=False,
        **kwargs,
    ) -> None:
        super().__init__(atomgroups, bin_width, surf_ids, **kwargs)
        self.cutoff = cutoff
        self.sfactor = sfactor
        self.perturbation = perturbation
        self.calc_unscaled = calc_unscaled

    def _prepare(self):
        super()._prepare()
        self.results.m_unscaled = np.zeros((self.nbins))
        self.results.mM_unscaled = np.zeros((self.nbins))
        self.results.M_unscaled = 0.0
        self.results.M2_unscaled = 0.0

    def _single_frame(self):
        super()._single_frame()

        if self.calc_unscaled:
            # charge density [e/A^3]
            rho, bin_edges = np.histogram(
                self.atoms.positions[:, 2], bins=bin_edges, weights=self.atoms.charges
            )
            bin_volumes = np.diff(bin_edges) * self.cross_area
            rho /= bin_volumes
            m = -integrate.cumulative_trapezoid(rho, self.bins, initial=0)
            self.results.m_unscaled += m

            # M
            M = np.sum(m * bin_volumes)
            self.results.M_unscaled += M
            self.results.M2_unscaled += M**2
            self.results.mM_unscaled += m * M

    def _conclude(self):
        super()._conclude()
        if self.calc_unscaled:
            self.results.m_unscaled /= self.n_frames
            self.results.mM_unscaled /= self.n_frames
            self.results.M_unscaled /= self.n_frames
            self.results.M2_unscaled /= self.n_frames

            x_fluct = (
                self.results.mM_unscaled
                - self.results.m_unscaled * self.results.M_unscaled
            )
            M_fluct = (
                self.results.M2_unscaled
                - self.results.M_unscaled * self.results.M_unscaled
            )
            self.results.inveps_unscaled = 1 - x_fluct / (
                self.const + M_fluct / self.results.volume
            )

    def _calc_rho(self):
        z = self.atoms.positions[:, self.axis]
        # scale charges of chemisorbed water
        atype_mask = self.atoms.types == "O"
        z_mask = (z <= (self._ts_z_lo + self.cutoff)) | (
            z >= (self._ts_z_hi - self.cutoff)
        )
        sel_O_ids = np.nonzero(atype_mask * z_mask)[0]
        sel_ids = np.sort(np.concatenate([sel_O_ids, sel_O_ids + 1, sel_O_ids + 2]))
        charges = self.atoms.charges.copy()
        charges[sel_ids] *= self.sfactor
        # add pert in charge and keep electroneutrality
        perturbation = np.random.uniform(
            low=-self.perturbation, high=self.perturbation, size=len(sel_ids)
        )
        perturbation -= perturbation.mean()
        charges[sel_ids] += perturbation

        # charge density [e/A^3]
        rho, bin_edges = np.histogram(z, bins=self.bin_edges, weights=charges)
        return rho


class NewAdInverseDielectricConstant(InverseDielectricConstant):
    def __init__(
        self,
        atomgroups,
        bin_width,
        surf_ids,
        cutoff=2.7,
        mu_ad=0.26,
        calc_unscaled=False,
        **kwargs,
    ) -> None:
        super().__init__(atomgroups, bin_width, surf_ids, **kwargs)
        self.cutoff = cutoff
        self.mu_ad = mu_ad
        self.calc_unscaled = calc_unscaled

    def _prepare(self):
        super()._prepare()
        self.results.m_unscaled = np.zeros((self.nbins))
        self.results.mM_unscaled = np.zeros((self.nbins))
        self.results.M_unscaled = 0.0
        self.results.M2_unscaled = 0.0

    def _single_frame(self):
        super()._single_frame()

        if self.calc_unscaled:
            # charge density [e/A^3]
            rho, bin_edges = np.histogram(
                self.atoms.positions[:, 2], bins=bin_edges, weights=self.atoms.charges
            )
            bin_volumes = np.diff(bin_edges) * self.cross_area
            rho /= bin_volumes
            m = -integrate.cumulative_trapezoid(rho, self.bins, initial=0)
            self.results.m_unscaled += m

            # M
            M = np.sum(m * bin_volumes)
            self.results.M_unscaled += M
            self.results.M2_unscaled += M**2
            self.results.mM_unscaled += m * M

    def _conclude(self):
        super()._conclude()
        if self.calc_unscaled:
            self.results.m_unscaled /= self.n_frames
            self.results.mM_unscaled /= self.n_frames
            self.results.M_unscaled /= self.n_frames
            self.results.M2_unscaled /= self.n_frames

            x_fluct = (
                self.results.mM_unscaled
                - self.results.m_unscaled * self.results.M_unscaled
            )
            M_fluct = (
                self.results.M2_unscaled
                - self.results.M_unscaled * self.results.M_unscaled
            )
            self.results.inveps_unscaled = 1 - x_fluct / (
                self.const + M_fluct / self.results.volume
            )

    def _calc_rho(self):
        z = self.atoms.positions[:, self.axis]
        # count number of chemisorbed water
        atype_mask = self.atoms.types == "O"
        charges = self.atoms.charges.copy()
        # lower surface
        z_mask = z <= (self._ts_z_lo + self.cutoff)
        n_wat = np.count_nonzero(atype_mask * z_mask)
        charges = np.append(charges, [-self.mu_ad * n_wat, self.mu_ad * n_wat])
        # upper surface
        z_mask = z >= (self._ts_z_hi - self.cutoff)
        n_wat = np.count_nonzero(atype_mask * z_mask)
        charges = np.append(charges, [-self.mu_ad * n_wat, self.mu_ad * n_wat])
        z = np.append(
            z,
            [
                self._ts_z_lo + 1,
                self._ts_z_lo + 2,
                self._ts_z_hi - 1,
                self._ts_z_hi - 2,
            ],
        )
        # charge density [e/A^3]
        rho, bin_edges = np.histogram(z, bins=self.bin_edges, weights=charges)
        return rho


class GaussianInverseDielectricConstant(InverseDielectricConstant):
    def __init__(
        self,
        atomgroups,
        bin_width,
        surf_ids,
        axis: int = 2,
        temperature=330,
        img_plane=0,
        oxygen_sigma=0.669,
        hydrogen_sigma=0.371,
        make_whole=False,
        dimensions=None,
        verbose=True,
    ) -> None:
        super().__init__(
            atomgroups,
            bin_width,
            surf_ids,
            axis,
            temperature,
            img_plane,
            make_whole,
            dimensions,
            verbose,
        )
        self.oxygen_sigma = oxygen_sigma
        self.hydrogen_sigma = hydrogen_sigma

    def _calc_rho(self):
        """
        rho: charge density [e/A^3]
        """
        sigma = np.zeros((len(self.atoms)))
        # water only
        coords = self.atoms.positions[:, self.axis].reshape(-1, 1)
        charges = self.atoms.charges
        bin_edges = np.reshape(self.bin_edges, (1, -1))
        sigma[charges > 0] = self.hydrogen_sigma
        sigma[charges < 0] = self.oxygen_sigma
        # nat * (nbins + 1)
        rho = charges.reshape(-1, 1) * gaussian_int(
            bin_edges, coords, sigma.reshape(-1, 1)
        )
        # nat * nbins
        rho = np.diff(rho, axis=1)
        return np.sum(rho, axis=0)


class DeepWannierInverseDielectricConstant(InverseDielectricConstant):
    def __init__(
        self,
        atomgroups,
        bin_width,
        surf_ids,
        model,
        symbols,
        **kwargs,
    ) -> None:
        from deepmd.infer import DeepDipole

        super().__init__(
            atomgroups,
            bin_width,
            surf_ids,
            **kwargs,
        )
        self.model = DeepDipole(model)
        self.type_map = self.model.tmap

        assert len(symbols) == len(self.universe.atoms)
        symbols = np.array(symbols)[self.atoms.indices]
        self.atype = np.zeros(len(self.atoms), dtype=np.int32)
        for ii, _atype in enumerate(self.type_map):
            self.atype[symbols == _atype] = ii

        charges = np.zeros_like(symbols)
        charges[symbols == "O"] = 6.0
        charges[symbols == "H"] = 1.0
        self.charges = np.concatenate(
            [charges, np.full_like((np.count_nonzero(symbols == "O")), -8.0)]
        )

        self.symbols = symbols

    def _calc_rho(self):
        """
        rho: charge density [e/A^3]
        """
        ion_coords = self.atoms.positions
        atomic_dipole = self._dp_eval(ion_coords)
        wannier_coords = ion_coords[self.symbols == "O"] + atomic_dipole
        extended_coords = np.concatenate([ion_coords, wannier_coords], axis=0)
        rho, bin_edges = np.histogram(
            extended_coords[:, self.axis], bins=self.bin_edges, weights=self.charges
        )
        return rho

    def _dp_eval(self, coord):
        cell = geometry.cell.cellpar_to_cell(self._ts.dimensions)
        dipole = self.model.eval(
            coord.reshape(1, -1),
            cell.reshape(1, 9),
            self.atype,
        )
        return dipole.reshape(-1, 3)


class DeepWannierGaussianInverseDielectricConstant(
    DeepWannierInverseDielectricConstant
):
    def __init__(
        self,
        atomgroups,
        bin_width,
        surf_ids,
        model,
        symbols,
        **kwargs,
    ) -> None:
        super().__init__(
            atomgroups,
            bin_width,
            surf_ids,
            model,
            symbols,
            **kwargs,
        )

        coeff = constants.physical_constants["Bohr radius"][0] / constants.angstrom
        wc_spread = 1.885748409412253 # data from pure water DFT (+-4e-3)
        g_spread_dict = {
            "O": 0.244554 * coeff,
            "H": 0.200000 * coeff,
            "X": np.sqrt(wc_spread / 3) * coeff,
        }
        self.g_spread = np.concatenate(
            [
                np.array([g_spread_dict[_] for _ in self.symbols]),
                np.full((np.count_nonzero(self.symbols == "O")), g_spread_dict["X"]),
            ]
        )

    def _calc_rho(self):
        """
        rho: charge density [e/A^3]
        """
        ion_coords = self.atoms.positions
        atomic_dipole = self._dp_eval(ion_coords)
        wannier_coords = ion_coords[self.symbols == "O"] + atomic_dipole
        extended_coords = np.concatenate([ion_coords, wannier_coords], axis=0)

        coords = extended_coords[:, self.axis].reshape(-1, 1)
        bin_edges = np.reshape(self.bin_edges, (1, -1))

        # nat * (nbins + 1)
        rho = self.charges.reshape(-1, 1) * gaussian_int(
            bin_edges, coords, self.g_spread.reshape(-1, 1)
        )
        # nat * nbins
        rho = np.diff(rho, axis=1)
        return np.sum(rho, axis=0)


class CP2KGaussianInverseDielectricConstant(InverseDielectricConstant):
    def __init__(
        self,
        atomgroups,
        bin_width,
        surf_ids,
        axis: int = 2,
        temperature=330,
        img_plane=0,
        make_whole=False,
        dimensions=None,
        verbose=True,
    ) -> None:
        super().__init__(
            atomgroups,
            bin_width,
            surf_ids,
            axis,
            temperature,
            img_plane,
            make_whole,
            dimensions,
            verbose,
        )

    def _calc_rho(self):
        """
        rho: charge density [e/A^3]
        """
        oxygen_e_sigma = 0.384
        hydrogen_e_sigma = 0.382
        oxygen_n_sigma = 0.129
        hydrogen_n_sigma = 0.106
        oxygen_e_charge = -0.956
        hydrogen_e_charge = -0.110
        oxygen_n_charge = 0.884
        hydrogen_n_charge = 0.146

        e_sigma = np.zeros((len(self.atoms)))
        n_sigma = np.zeros((len(self.atoms)))
        e_charges = np.zeros((len(self.atoms)))
        n_charges = np.zeros((len(self.atoms)))
        # water only!
        coords = self.atoms.positions[:, self.axis].reshape(-1, 1)
        bin_edges = np.reshape(self.bin_edges, (1, -1))
        # set hydrogen param
        mask = self.atoms.charges > 0
        e_sigma[mask] = hydrogen_e_sigma
        n_sigma[mask] = hydrogen_n_sigma
        e_charges[mask] = hydrogen_e_charge
        n_charges[mask] = hydrogen_n_charge
        # set oxygen param
        mask = self.atoms.charges < 0
        e_sigma[mask] = oxygen_e_sigma
        n_sigma[mask] = oxygen_n_sigma
        e_charges[mask] = oxygen_e_charge
        n_charges[mask] = oxygen_n_charge

        # nat * (nbins + 1)
        rho_e = e_charges.reshape(-1, 1) * gaussian_int(
            bin_edges, coords, e_sigma.reshape(-1, 1)
        )
        rho_n = n_charges.reshape(-1, 1) * gaussian_int(
            bin_edges, coords, n_sigma.reshape(-1, 1)
        )
        rho = rho_n + rho_e
        # nat * nbins
        rho = np.diff(rho, axis=1)
        return np.sum(rho, axis=0)


class DeepWannierDielectricConstant(DielectricConstant):
    """
    Analysis class for bulk water + DP potential
    """

    def __init__(
        self,
        atomgroups,
        temperature=330.0,
        make_whole=False,
        model="graph.pb",
        symbols=None,
        **kwargs,
    ):
        from deepmd.infer import DeepDipole
        
        super().__init__(atomgroups, temperature, make_whole, **kwargs)
        self.universe = atomgroups.universe
        
        self.model = DeepDipole(model)
        self.type_map = self.model.tmap

        assert len(symbols) == len(self.universe.atoms)
        symbols = np.array(symbols)[self.atomgroup.indices]
        self.atype = np.zeros(len(self.atomgroup), dtype=np.int32)
        for ii, _atype in enumerate(self.type_map):
            self.atype[symbols == _atype] = ii

        charges = np.zeros_like(symbols)
        charges[symbols == "O"] = 6.0
        charges[symbols == "H"] = 1.0
        self.charges = np.concatenate(
            [charges, np.full_like((np.count_nonzero(symbols == "O")), -8.0)]
        )

    def _single_frame(self):
        if self.make_whole:
            self.atomgroup.unwrap()

        self.volume += self.atomgroup.universe.trajectory.ts.volume
        
        ion_coords = self.atomgroup.positions
        atomic_dipole = self._dp_eval(ion_coords)
        wannier_coords = ion_coords[self.symbols == "O"] + atomic_dipole
        extended_coords = np.concatenate([ion_coords, wannier_coords], axis=0)
        
        M = np.dot(self.charges, extended_coords)
        
        self.results.M += M
        self.results.M2 += M * M

    def _dp_eval(self, coord):
        cell = geometry.cell.cellpar_to_cell(self._ts.dimensions)
        dipole = self.model.eval(
            coord.reshape(1, -1),
            cell.reshape(1, 9),
            self.atype,
        )
        return dipole.reshape(-1, 3)
