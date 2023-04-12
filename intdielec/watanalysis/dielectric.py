import numpy as np
# import multiprocessing as mp

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.analysis.dielectric import DielectricConstant
from MDAnalysis.units import constants, convert
# from pmda.parallel import ParallelAnalysisBase

from ..utils.unit import *
from scipy import integrate


class InverseDielectricConstant(AnalysisBase):
    def __init__(self,
                 atomgroups,
                 bin_edges,
                 axis: int = 2,
                 temperature=330,
                 img_plane=0.,
                 make_whole=True,
                 verbose=False,
                 **kwargs) -> None:
        self.universe = atomgroups.universe
        super().__init__(self.universe.trajectory, verbose)

        self.atoms = atomgroups
        self.bin_width = bin_edges[1] - bin_edges[0]
        self.bins = (bin_edges[1:] + bin_edges[:-1]) / 2
        self.nbins = len(bin_edges) - 1
        self.axis = axis
        self.temperature = temperature
        self.img_plane = img_plane
        self.make_whole = make_whole
        self.kwargs = kwargs

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
        self.volume = 0.

    def _single_frame(self):
        if self.make_whole:
            self.atoms.unwrap()

        ave_axis = np.delete(np.arange(3), self.axis)
        ts_area = self._ts.dimensions[ave_axis[0]] * self._ts.dimensions[
            ave_axis[1]]

        # get refs
        # ts_volume = self._ts.volume
        surf_ids = self.kwargs["surf_ids"]
        # print(surf_ids)
        z_lo = np.mean(self.atoms.positions[surf_ids[0]][:, 2])
        z_hi = np.mean(self.atoms.positions[surf_ids[1]][:, 2])
        bin_edges = np.linspace(z_lo, z_hi,
                                int((z_hi - z_lo) / self.bin_width) + 1)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2.

        # charge density [e/A^3]
        z = self.atoms.positions[:, self.axis]
        rho, bin_edges = np.histogram(z,
                                      bins=bin_edges,
                                      weights=self.atoms.charges)
        bin_volumes = np.diff(bin_edges) * ts_area
        rho /= bin_volumes
        # m
        _m = -integrate.cumulative_trapezoid(rho, self.bins, initial=0)
        # M
        M = np.dot(self.atoms.charges, self.atoms.positions)[self.axis]
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
        self.volume += ts_volume

    def _conclude(self):
        self.results.m /= (self.n_frames * 2)
        self.results.mM /= (self.n_frames * 2)
        self.results.M /= self.n_frames
        self.results.M2 /= self.n_frames
        self.volume /= self.n_frames

        x_fluct = self.results.mM - self.results.m * self.results.M
        M_fluct = self.results.M2 - self.results.M * self.results.M

        self.results.eps = self.results.fluct / (
            convert(constants["Boltzman_constant"], "kJ/mol", "eV") *
            self.temperature * self.volume * constants["electric_constant"])

        const = convert(
            constants["Boltzman_constant"], "kJ/mol",
            "eV") * self.temperature * constants["electric_constant"] / 3.
        self.results.inveps = 1 - x_fluct / (const + M_fluct / self.volume)


class ParallelInverseDielectricConstant(InverseDielectricConstant):
    def __init__(self,
                 universe,
                 bins,
                 axis="z",
                 temperature=330,
                 make_whole=True,
                 verbose=False,
                 **kwargs) -> None:
        super().__init__(universe, bins, axis, temperature, make_whole,
                         verbose, **kwargs)
        #parallel value initial
        self.para = None
        self._para_region = None

    def _conclude(self):
        pass

    def _parallel_init(self, *args, **kwargs):
        start = self._para_region.start
        stop = self._para_region.stop
        step = self._para_region.step
        self._setup_frames(self._trajectory, start, stop, step)
        self._prepare()

    def run(self, start=None, stop=None, step=None, verbose=None):

        #self._trajectory._reopen()
        if verbose == True:
            print(" ", end='')
        super().run(start, stop, step, verbose)

        if self.para:
            block_result = self._para_block_result()
            if block_result == None:
                raise ValueError(
                    "in parallel, block result has not been defined or no data output!"
                )
            #logger.info("block_anal finished.")
            return block_result

    def _para_block_result(self):
        return [
            self.results.m, self.results.mM, self.results.M, self.results.M2,
            self.volume
        ]

    def _parallel_conclude(self, rawdata):
        # set attributes for further analysis
        method_attr = rawdata[-1]
        del rawdata[-1]
        self.start = method_attr[0]
        self.stop = method_attr[1]
        self.step = method_attr[2]
        self.frames = np.arange(self.start, self.stop, self.step)
        self.n_frames = len(self.frames)

        self.results["m"] = np.zeros((self.nbins))
        self.results["mM"] = np.zeros((self.nbins))
        self.results["M"] = 0
        self.results["M2"] = 0
        self.volume = 0

        for single_data in rawdata:
            self.results["m"] += single_data[0]
            self.results["mM"] += single_data[1]
            self.results["M"] += single_data[2]
            self.results["M2"] += single_data[3]
            self.volume += single_data[4]

        self.results["m"] /= self.n_frames
        self.results["mM"] /= self.n_frames
        self.results["M"] /= self.n_frames
        self.results["M2"] /= self.n_frames
        self.volume /= self.n_frames

        x_fluct = self.results["mM"] - self.results["m"] * self.results["M"]
        M_fluct = self.results["M2"] - self.results["M"] * self.results["M"]
        const = convert(
            constants["Boltzman_constant"], "kJ/mol",
            "eV") * self.temperature * constants["electric_constant"]
        self.results["inveps"] = 1 - x_fluct / (const + M_fluct / self.volume)

        return "FINISH PARA CONCLUDE"
