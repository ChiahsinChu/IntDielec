from scipy import stats
from ase.geometry.cell import cellpar_to_cell

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import \
    HydrogenBondAnalysis
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import capped_distance, minimize_vectors

from .base import ReWeightingOP

from ..exts.toolbox.toolbox.utils import *
from ..exts.toolbox.toolbox.utils.unit import *
from ..exts.toolbox.toolbox.utils.utils import calc_water_density
from ..utils.mda import make_selection, make_selection_two


class WatCoverage(AnalysisBase):
    def __init__(self, universe, verbose=False, **kwargs):
        select = make_selection(**kwargs)
        # print("selection: ", select)
        self.universe = universe
        trajectory = universe.trajectory
        super().__init__(trajectory, verbose=verbose)
        self.n_frames = len(trajectory)
        self.ag = universe.select_atoms(select, updating=True)

    def _prepare(self):
        # placeholder for water z
        self.n_water = np.zeros((self.n_frames), dtype=np.int32)

    def _single_frame(self):
        self.n_water[self._frame_index] = len(self.ag)

    def _conclude(self):
        return self.n_water


class WatDensity(AnalysisBase):
    def __init__(self,
                 universe,
                 bin_edges,
                 surf_ids,
                 sel_water="name O",
                 axis: int = 2,
                 verbose=False,
                 **kwargs):

        super().__init__(universe.trajectory, verbose, **kwargs)
        self.universe = universe
        self.water = universe.select_atoms(sel_water)

        self.bin_width = bin_edges[1] - bin_edges[0]
        self.bins = (bin_edges[1:] + bin_edges[:-1]) / 2
        self.nbins = len(bin_edges) - 1
        self.surf_ids = surf_ids
        self.axis = axis

    def _prepare(self):
        self.result = np.zeros((self.nbins))

    def _single_frame(self):
        # get refs
        z = self._ts.positions[:, self.axis]
        z_ave = [np.mean(z[self.surf_ids[0]]), np.mean(z[self.surf_ids[1]])]
        z_lo = np.min(z_ave)
        z_hi = np.max(z_ave)
        
        raw_z = self.water.positions[:, self.axis]

        bin_edges = np.linspace(
            0., self._ts.dimensions[self.axis],
            int(self._ts.dimensions[self.axis] / self.bin_width) + 1)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2.
        n_wat, bin_edges = np.histogram(raw_z, bins=bin_edges)
        bin_volumes = np.diff(bin_edges) * self.cross_area
        rho = calc_water_density(n_wat, bin_volumes)

        self.result += np.interp(self.bins + z_lo, bins, rho)
        self.result += np.flip(np.interp(np.sort(z_hi - self.bins), bins, rho))

    def _conclude(self):
        self.result /= (self.n_frames * 2)

    def save(self, fname="water_density.txt"):
        np.savetxt(fname,
                   np.transpose([self.bins, self.result]),
                   header="x[A] rho[g/cm^3]")

    @property
    def cross_area(self):
        ave_axis = np.delete(np.arange(3), self.axis)
        cell = cellpar_to_cell(self._ts.dimensions)
        ts_area = np.linalg.norm(np.cross(cell[ave_axis[0]], cell[ave_axis[1]]))
        return ts_area


class ReWeightingWatDensity(WatDensity, ReWeightingOP):
    def __init__(self, 
                 universe, 
                 bin_edges, 
                 surf_ids, 
                 sel_water="name O", 
                 axis: int = 2, 
                 verbose=False, 
                 **kwargs):
        super().__init__(universe, bin_edges, surf_ids, 
                         sel_water, axis, verbose, **kwargs)
        
    def _prepare(self):
        super()._prepare()
        self.weights = self.calc_weights()
    
    def _single_frame(self):
        # get refs
        z = self._ts.positions[:, self.axis]
        z_ave = [np.mean(z[self.surf_ids[0]]), np.mean(z[self.surf_ids[1]])]
        z_lo = np.min(z_ave)
        z_hi = np.max(z_ave)
        
        raw_z = self.water.positions[:, self.axis]

        bin_edges = np.linspace(
            0., self._ts.dimensions[self.axis],
            int(self._ts.dimensions[self.axis] / self.bin_width) + 1)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2.
        n_wat, bin_edges = np.histogram(raw_z, bins=bin_edges)
        bin_volumes = np.diff(bin_edges) * self.cross_area
        rho = calc_water_density(n_wat, bin_volumes)

        ts_result = np.interp(self.bins + z_lo, bins, rho)
        ts_result += np.flip(np.interp(np.sort(z_hi - self.bins), bins, rho))
        ts_result /= 2.
        weight = self.weights[self._frame_index]
        self.result += (ts_result * weight)
        
    def _conclude(self):
        self.result /= np.sum(self.weights)


class AngularDistribution(AnalysisBase):
    def __init__(self,
                 universe,
                 nbins=50,
                 axis: int = 2,
                 updating=True,
                 verbose=False,
                 **kwargs):
        trajectory = universe.trajectory
        super().__init__(trajectory, verbose=verbose)

        select = make_selection_two(**kwargs)
        self.universe = universe
        self.updating = updating
        self.ags = self._make_selections(select)
        self.nbins = nbins
        self.axis = axis

    def _prepare(self):
        self.ts_cosOH = []
        self.ts_cosHH = []
        self.ts_cosD = []

    def _single_frame(self):
        # NOTE: be careful! for OHH only yet!
        axis = self.axis

        cosOH, cosHH, cosD = self._getCosTheta(self.ags[0], axis)
        self.ts_cosOH.extend(cosOH.tolist())
        self.ts_cosHH.extend(cosHH.tolist())
        self.ts_cosD.extend(cosD.tolist())
        cosOH, cosHH, cosD = self._getCosTheta(self.ags[1], axis)
        self.ts_cosOH.extend((-cosOH).tolist())
        self.ts_cosHH.extend((-cosHH).tolist())
        self.ts_cosD.extend((-cosD).tolist())

    def _conclude(self):
        self.results = {}

        thetaOH = np.arccos(self.ts_cosOH) / np.pi * 180
        thetaHH = np.arccos(self.ts_cosHH) / np.pi * 180
        thetaD = np.arccos(self.ts_cosD) / np.pi * 180

        cos_hist_interval = np.linspace(-1., 1., self.nbins)
        theta_hist_interval = np.linspace(0., 180., self.nbins)

        hist_cosOH = np.histogram(self.ts_cosOH,
                                  cos_hist_interval,
                                  density=True)
        hist_cosHH = np.histogram(self.ts_cosHH,
                                  cos_hist_interval,
                                  density=True)
        hist_cosD = np.histogram(self.ts_cosD, cos_hist_interval, density=True)
        hist_OH = np.histogram(thetaOH, theta_hist_interval, density=True)
        hist_HH = np.histogram(thetaHH, theta_hist_interval, density=True)
        hist_D = np.histogram(thetaD, theta_hist_interval, density=True)

        for label in ['cosOH', 'cosHH', 'cosD', 'OH', 'HH', 'D']:
            output = locals()['hist_%s' % label]
            self.results[label] = np.transpose(
                np.concatenate(
                    ([output[1][:-1] + (output[1][1] - output[1][0]) / 2],
                     [output[0]])))

    def _getCosTheta(self, ag, axis):
        ts_positions = ag.positions
        ts_p_O = ts_positions[::3]
        ts_p_H1 = ts_positions[1::3]
        ts_p_H2 = ts_positions[2::3]

        vec_OH_0 = minimize_vectors(vectors=ts_p_H1 - ts_p_O,
                                    box=self._ts.dimensions)
        vec_OH_1 = minimize_vectors(vectors=ts_p_H2 - ts_p_O,
                                    box=self._ts.dimensions)
        cosOH = vec_OH_0[:, axis] / np.linalg.norm(vec_OH_0, axis=-1)
        # self.ts_cosOH.extend(cosOH.tolist())
        cosOH = np.append(
            cosOH, vec_OH_1[:, axis] / np.linalg.norm(vec_OH_1, axis=-1))
        # self.ts_cosOH.extend(cosOH.tolist())

        vec_HH = ts_p_H1 - ts_p_H2
        cosHH = vec_HH[:, axis] / np.linalg.norm(vec_HH, axis=-1)
        # self.ts_cosHH.extend(cosHH.tolist())

        vec_D = vec_OH_0 + vec_OH_1
        cosD = vec_D[:, axis] / np.linalg.norm(vec_D, axis=-1)
        # self.ts_cosD.extend(cosD.tolist())

        return cosOH, cosHH, cosD

    def _make_selections(self, l_selection_str):
        selection = []
        for sel in l_selection_str:
            sel_ag = self.universe.select_atoms(sel, updating=self.updating)
            # TODO: check why it does not work
            sel_ag.unwrap()
            selection.append(sel_ag)
        return selection


# TODO: to be checked >>>>>>>>>>>>>


class HBA(HydrogenBondAnalysis):
    def __init__(self,
                 universe,
                 donors_sel=None,
                 hydrogens_sel=None,
                 acceptors_sel=None,
                 between=None,
                 d_h_cutoff=1.2,
                 d_a_cutoff=3,
                 d_h_a_angle_cutoff=150,
                 update_acceptors=False,
                 update_donors=False):
        self.update_acceptors = update_acceptors
        self.update_donors = update_donors
        update_selection = (update_donors | update_acceptors)
        super().__init__(universe, donors_sel, hydrogens_sel, acceptors_sel,
                         between, d_h_cutoff, d_a_cutoff, d_h_a_angle_cutoff,
                         update_selection)

    def _prepare(self):
        self.results.hbonds = [[], [], [], [], [], []]

        # Set atom selections if they have not been provided
        if not self.acceptors_sel:
            self.acceptors_sel = self.guess_acceptors()
        if not self.hydrogens_sel:
            self.hydrogens_sel = self.guess_hydrogens()

        # Select atom groups
        self._acceptors = self.u.select_atoms(self.acceptors_sel,
                                              updating=self.update_acceptors)
        self._donors, self._hydrogens = self._get_dh_pairs()

    def _get_dh_pairs(self):
        """Finds donor-hydrogen pairs.

        Returns
        -------
        donors, hydrogens: AtomGroup, AtomGroup
            AtomGroups corresponding to all donors and all hydrogens. AtomGroups are ordered such that, if zipped, will
            produce a list of donor-hydrogen pairs.
        """

        # If donors_sel is not provided, use topology to find d-h pairs
        if not self.donors_sel:
            # We're using u._topology.bonds rather than u.bonds as it is a million times faster to access.
            # This is because u.bonds also calculates properties of each bond (e.g bond length).
            # See https://github.com/MDAnalysis/mdanalysis/issues/2396#issuecomment-596251787
            if not (hasattr(self.u._topology, 'bonds')
                    and len(self.u._topology.bonds.values) != 0):
                raise NoDataError(
                    'Cannot assign donor-hydrogen pairs via topology as no bond information is present. '
                    'Please either: load a topology file with bond information; use the guess_bonds() '
                    'topology guesser; or set HydrogenBondAnalysis.donors_sel so that a distance cutoff '
                    'can be used.')

            hydrogens = self.u.select_atoms(self.hydrogens_sel)
            donors = sum(h.bonded_atoms[0] for h in hydrogens) if hydrogens \
                else AtomGroup([], self.u)

        # Otherwise, use d_h_cutoff as a cutoff distance
        else:
            hydrogens = self.u.select_atoms(self.hydrogens_sel)
            donors = self.u.select_atoms(self.donors_sel,
                                         updating=self.update_donors)
            donors_indices, hydrogen_indices = capped_distance(
                donors.positions,
                hydrogens.positions,
                max_cutoff=self.d_h_cutoff,
                box=self.u.dimensions,
                return_distances=False).T

            donors = donors[donors_indices]
            hydrogens = hydrogens[hydrogen_indices]

        return donors, hydrogens


class Density1DAnalysis(AnalysisBase):
    """
    Parameters
    ----------
    universe: AtomGroup Object
        tbc
    water_sel: List or Array
        [a, b, c, alpha, beta, gamma]
    dim: int (2)
        
    delta: float (0.1)
        tbc
    """
    def __init__(self,
                 universe,
                 water_sel='name O',
                 dim=2,
                 delta=0.1,
                 mass=18.015):
        super().__init__(universe.trajectory)
        self._cell = universe.dimensions
        self._dim = dim
        self._delta = delta
        self._mass = mass
        self._nwat = len(universe.select_atoms(water_sel))
        self._O_ids = universe.select_atoms(water_sel).indices

        # check cell
        if self._cell is None:
            raise AttributeError('Cell parameters should be set.')

        #parallel value initial
        self.para = None
        self._para_region = None

    def _prepare(self):
        # cross area along the selected dimension
        dims = self._cell[:3]
        dims = np.delete(dims, self._dim)
        self.cross_area = dims[0] * dims[1] * np.sin(
            self._cell[self._dim + 3] / 180 * np.pi)
        # placeholder
        self.all_coords = np.zeros((self.n_frames, self._nwat),
                                   dtype=np.float32)

    def _single_frame(self):
        ts_coord = self._ts.positions[self._O_ids].T[self._dim]
        np.copyto(self.all_coords[self._frame_index], ts_coord)

    def _conclude(self):
        bins = np.arange(0, self._cell[self._dim], self._delta)
        grids, density = self._get_density(self.all_coords, bins)
        self.results = {}
        self.results['grids'] = grids
        self.results['density'] = density

    def _get_density(self, coords, bins):
        density, grids = np.histogram(coords, bins=bins)
        grids = grids[:-1] + self._delta / 2
        density = (density / constants.Avogadro * self._mass) / (
            self.cross_area * self._delta *
            (constants.angstrom / constants.centi)**3) / self.n_frames
        return grids, density

    def _parallel_init(self, *args, **kwargs):

        start = self._para_region.start
        stop = self._para_region.stop
        step = self._para_region.step
        self._setup_frames(self._trajectory, start, stop, step)
        self._prepare()

    def run(self, start=None, stop=None, step=None, verbose=None):
        if verbose == True:
            print(" ", end='')
        super().run(start, stop, step, verbose)

        if self.para:
            block_result = self._para_block_result()
            if block_result == None:
                raise ValueError(
                    "in parallel, block result has not been defined or no data output!"
                )
            return block_result

    def to_file(self, output_file):
        output = np.concatenate(
            ([self.results['grids']], [self.results['density']]), axis=0)
        output = np.transpose(output)
        if os.path.splitext(output_file)[-1][1:] == "npy":
            np.save(output_file, output)
        else:
            np.savetxt(output_file, output)

    def _para_block_result(self, ):
        return self.results

    def _parallel_conclude(self, rawdata):
        method_attr = rawdata[-1]
        del rawdata[-1]
        self.start = method_attr[0]
        self.stop = method_attr[1]
        self.step = method_attr[2]
        self.frames = np.arange(self.start, self.stop, self.step)

        self.results = {}
        density = []
        for single_data in rawdata:
            density.append(single_data['density'])
        density = np.mean(density, axis=0)
        self.results['grids'] = single_data['grids']
        self.results['density']
        return "FINISH PARA CONCLUDE"


class InterfaceWatDensity(Density1DAnalysis):
    """
    TBC
    """
    def __init__(self,
                 universe,
                 water_sel='name O',
                 dim=2,
                 delta=0.1,
                 **kwargs):
        super().__init__(universe, water_sel, dim, delta, mass=18.015)

        self._surf_ids = kwargs.get('surf_ids', None)
        if self._surf_ids is None:
            # if no surf ids provided
            slab_sel = kwargs.get('slab_sel', None)
            self._surf_natoms = kwargs.get('surf_natoms', None)
            if slab_sel is not None and self._surf_natoms is not None:
                self._slab_ids = universe.select_atoms(slab_sel).indices
            else:
                raise AttributeError(
                    'slab_sel and surf_natoms should be provided in the absence of surf_ids'
                )

    def _prepare(self):
        super()._prepare()
        # placeholder for surface coords
        self.surf_coords = np.zeros((self.n_frames, 2), dtype=np.float32)
        #print(self.surf_ids)

    def _single_frame(self):
        # save surface coords
        ts_surf_lo = self._ts.positions[self.surf_ids[0]].T[self._dim]
        ts_surf_hi = self._ts.positions[self.surf_ids[1]].T[self._dim]
        ts_surf_region = self._get_surf_region(ts_surf_lo, ts_surf_hi)
        np.copyto(self.surf_coords[self._frame_index], ts_surf_region)

        # save water coords w.r.t. lower surfaces
        _ts_coord = self._ts.positions[self._O_ids].T[self._dim]
        ts_coord = self._ref_water(_ts_coord, ts_surf_region[0])
        np.copyto(self.all_coords[self._frame_index], ts_coord)

    def _conclude(self):
        surf_lo_ave = self.surf_coords[:, 0].mean(axis=0)
        surf_hi_ave = self.surf_coords[:, 1].mean(axis=0)
        self._surf_space = surf_hi_ave - surf_lo_ave
        bins = np.arange(0, (self._surf_space + self._delta), self._delta)

        grids, density = self._get_density(self.all_coords, bins)
        self.results = {}
        self.results['grids'] = grids[:len(grids) // 2]
        self.results['density'] = (density + density[np.arange(
            len(density) - 1, -1, -1)])[:len(grids) // 2] / 2

    def _parallel_conclude(self, rawdata):
        method_attr = rawdata[-1]
        del rawdata[-1]
        self.start = method_attr[0]
        self.stop = method_attr[1]
        self.step = method_attr[2]
        self.frames = np.arange(self.start, self.stop, self.step)

        n_grids = 0
        for single_data in rawdata:
            if n_grids == 0 or len(single_data['grids']) < n_grids:
                n_grids = len(single_data['grids'])

        _density = []
        _grids = []
        for single_data in rawdata:
            _density.append(single_data['density'][:n_grids])
            _grids.append(single_data['grids'][:n_grids])
        self.results['grids'] = np.mean(_grids, axis=0)
        self.results['density'] = np.mean(_density, axis=0)

        return "FINISH PARA CONCLUDE"

    @property
    def surf_ids(self):
        """
        Get the indices of surface atoms
        """
        if self._surf_ids is None:
            from ase import Atoms
            slab = Atoms("H" + str(len(self._slab_ids)))
            slab.set_cell(self._cell)
            slab.set_pbc(True)
            pos = self._trajectory[0].positions[self._slab_ids]
            slab.set_positions(pos)
            slab.center(about=slab[0].position)
            slab.wrap()
            slab.center()

            coords = np.array([slab.positions[:, self._dim]])
            slab_ids = np.atleast_2d(self._slab_ids)
            data = np.concatenate((slab_ids, coords), axis=0)
            data = np.transpose(data)

            # sort from small coord to large coord
            data = data[data[:, 1].argsort()]
            upper_slab_ids = data[:self._surf_natoms][:, 0]
            upper_slab_ids.sort()
            lower_slab_ids = data[-self._surf_natoms:][:, 0]
            lower_slab_ids.sort()
            self._surf_ids = np.array([lower_slab_ids, upper_slab_ids],
                                      dtype=int)
        return self._surf_ids

    def _get_surf_region(self, _coord_lo, _coord_hi):
        """
        Return:
            numpy array of (coord_lo, coord_hi)
            coord_lo: average coord for lower surface
            coord_hi: average coord for upper surface
        """
        # fold all _coord_lo about _coord_lo.min()
        tmp = _coord_lo.min()
        for coord in _coord_lo:
            coord = coord + np.floor(
                (tmp - coord) / self._cell[self._dim]) * self._cell[self._dim]
        coord_lo = np.mean(_coord_lo)
        # fold all _coord_hi about _coord_hi.max()
        tmp = _coord_hi.max()
        for coord in _coord_hi:
            coord = coord + np.floor(
                (tmp - coord) / self._cell[self._dim]) * self._cell[self._dim]
        coord_hi = np.mean(_coord_hi)

        if coord_hi < coord_lo:
            coord_hi = coord_hi + self._cell[self._dim]

        return np.array([coord_lo, coord_hi], dtype=float)

    def _ref_water(self, water_coords, surf_lo):
        """
        water coord w.r.t. lower surfaces
        """
        for coord in water_coords:
            while coord < surf_lo:
                coord = coord + self._cell[self._dim]
        water_coords = water_coords - surf_lo
        return water_coords


class InterfaceWatOri(InterfaceWatDensity):
    """
    TBC
    """
    def __init__(self,
                 universe,
                 O_sel='name O',
                 H_sel='name H',
                 dim=2,
                 delta=0.1,
                 OH_cutoff=1.3,
                 update_pairs=False,
                 **kwargs):
        super().__init__(universe,
                         O_sel,
                         dim,
                         delta,
                         surf_ids=kwargs.get('surf_ids', None),
                         slab_sel=kwargs.get('slab_sel', None),
                         surf_natoms=kwargs.get('surf_natoms', None))
        self._H_ids = universe.select_atoms(H_sel).indices
        if len(self._H_ids) != self._nwat * 2:
            raise AttributeError('Only pure water has been supported yet.')
        self._update_pairs = update_pairs
        self.pairs = kwargs.get('pairs', None)
        self._OH_cutoff = OH_cutoff
        if self.pairs is None:
            self._get_OH_pairs(self._trajectory[0].positions[self._O_ids],
                               self._trajectory[0].positions[self._H_ids])

    def _prepare(self):
        super()._prepare()
        self.all_oris = np.zeros((self.n_frames, self._nwat), dtype=np.float32)

    def _single_frame(self):
        super()._single_frame()
        ts_O_coord = self._ts.positions[self._O_ids]
        ts_H_coord = self._ts.positions[self._H_ids]
        if self._update_pairs:
            self._get_pairs(ts_O_coord, ts_H_coord)
        ts_water_oris = self._get_water_ori(ts_O_coord, ts_H_coord)
        np.copyto(self.all_oris[self._frame_index], ts_water_oris)

    def _conclude(self):
        super()._conclude()
        all_coords = np.reshape(self.all_coords, -1)
        all_oris = np.reshape(self.all_oris, -1)
        bins = np.arange(0, (self._surf_space + self._delta), self._delta)
        water_cos, bin_edges, binnumber = stats.binned_statistic(
            x=all_coords, values=all_oris, bins=bins)
        water_cos = (water_cos - water_cos[np.arange(
            len(water_cos) - 1, -1, -1)])[:len(water_cos) // 2] / 2
        self.results['ori_dipole'] = water_cos * self.results['density']

    def _parallel_conclude(self, rawdata):
        method_attr = rawdata[-1]
        del rawdata[-1]
        self.start = method_attr[0]
        self.stop = method_attr[1]
        self.step = method_attr[2]
        self.frames = np.arange(self.start, self.stop, self.step)

        n_grids = 0
        for single_data in rawdata:
            if n_grids == 0 or len(single_data['grids']) < n_grids:
                n_grids = len(single_data['grids'])

        _density = []
        _grids = []
        _ori_dipole = []
        for single_data in rawdata:
            _density.append(single_data['density'][:n_grids])
            _grids.append(single_data['grids'][:n_grids])
            _ori_dipole.append(single_data['ori_dipole'][:n_grids])
        self.results['grids'] = np.mean(_grids, axis=0)
        self.results['density'] = np.mean(_density, axis=0)
        self.results['ori_dipole'] = np.mean(_ori_dipole, axis=0)

        return "FINISH PARA CONCLUDE"

    def _get_water_ori(self, O_coord, H_coord):
        """
        TBC
        """
        # get all OH bond vectors
        water_oris = np.zeros((self._nwat))
        OH_vecs = np.zeros((len(self.pairs[0]), 3))
        minimize_vectors(O_coord[self.pairs[0]],
                         H_coord[self.pairs[1]],
                         box=self._cell,
                         result=OH_vecs)
        # get the number of H assigned to each O
        ids, counts = np.unique(self.pairs[0], return_counts=True)
        # calculate the cosine of dipole
        for ii, id, count in zip(np.arange(self._nwat), ids, counts):
            tmp_vec = np.sum(OH_vecs[id:(id + count)], axis=0)
            water_oris[ii] = tmp_vec[self._dim] / np.linalg.norm(tmp_vec)
        return water_oris

    def _get_OH_pairs(self, O_coords, H_coords):
        O_ids, H_ids = capped_distance(O_coords,
                                       H_coords,
                                       max_cutoff=self._OH_cutoff,
                                       box=self._cell,
                                       return_distances=False).T
        self.pairs = [O_ids, H_ids]
