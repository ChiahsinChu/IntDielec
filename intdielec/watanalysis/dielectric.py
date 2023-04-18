import logging
import time

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
        logging.info("Start calculation")

        if not hasattr(self.atoms, "charges"):
            raise NoDataError("No charges defined given atomgroup.")

        if not np.allclose(
                self.atoms.total_charge(compound='fragments'), 0.0, atol=1E-5):
            raise NotImplementedError("Analysis for non-neutral systems or"
                                      " systems with free charges are not"
                                      " available.")

        # self.results.m_lo = np.zeros((self.nbins))
        # self.results.mM_lo = np.zeros((self.nbins))
        # self.results.m_hi = np.zeros((self.nbins))
        # self.results.mM_hi = np.zeros((self.nbins))
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

        logging.info("Number of frames: %s" % self.n_frames)
        logging.info("End calculation")


class AdInverseDielectricConstant(InverseDielectricConstant):
    def __init__(self,
                 atomgroups,
                 bin_edges,
                 surf_ids,
                 cutoff=2.7,
                 sfactor=2,
                 **kwargs) -> None:
        super().__init__(atomgroups, bin_edges, surf_ids, **kwargs)
        self.cutoff = cutoff
        self.sfactor = sfactor

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
        sel_O_ids = np.nonzero(atype_mask * z_mask)
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


# class ParallelInverseDielectricConstant(InverseDielectricConstant):
#     def __init__(self,
#                  atomgroups,
#                  bin_edges,
#                  surf_ids,
#                  axis: int = 2,
#                  temperature=330,
#                  img_plane=0,
#                  make_whole=False,
#                  verbose=False) -> None:
#         super().__init__(atomgroups, bin_edges, surf_ids, axis, temperature,
#                          img_plane, make_whole, verbose)
#         #parallel value initial
#         self.para = None
#         self._para_region = None

#     def _conclude(self):
#         pass

#     def _parallel_init(self, *args, **kwargs):
#         start = self._para_region.start
#         stop = self._para_region.stop
#         step = self._para_region.step
#         self._setup_frames(self._trajectory, start, stop, step)
#         self._prepare()

#     def _run(self, start=None, stop=None, step=None, verbose=None):

#         #self._trajectory._reopen()
#         if verbose == True:
#             print(" ", end='')
#         super().run(start, stop, step, verbose)

#         if self.para:
#             block_result = self._para_block_result()
#             if block_result == None:
#                 raise ValueError(
#                     "in parallel, block result has not been defined or no data output!"
#                 )
#             #logger.info("block_anal finished.")
#             return block_result

#     def _para_block_result(self):
#         return [
#             self.results.m, self.results.mM, self.results.M, self.results.M2,
#             self.volume
#         ]

#     def _parallel_conclude(self, rawdata):
#         # set attributes for further analysis
#         method_attr = rawdata[-1]
#         del rawdata[-1]
#         self.start = method_attr[0]
#         self.stop = method_attr[1]
#         self.step = method_attr[2]
#         self.frames = np.arange(self.start, self.stop, self.step)
#         self.n_frames = len(self.frames)

#         self.results["m"] = np.zeros((self.nbins))
#         self.results["mM"] = np.zeros((self.nbins))
#         self.results["M"] = 0
#         self.results["M2"] = 0
#         self.volume = 0

#         for single_data in rawdata:
#             self.results["m"] += single_data[0]
#             self.results["mM"] += single_data[1]
#             self.results["M"] += single_data[2]
#             self.results["M2"] += single_data[3]
#             self.volume += single_data[4]

#         self.results["m"] /= self.n_frames
#         self.results["mM"] /= self.n_frames
#         self.results["M"] /= self.n_frames
#         self.results["M2"] /= self.n_frames
#         self.volume /= self.n_frames

#         x_fluct = self.results["mM"] - self.results["m"] * self.results["M"]
#         M_fluct = self.results["M2"] - self.results["M"] * self.results["M"]
#         const = convert(
#             constants["Boltzman_constant"], "kJ/mol",
#             "eV") * self.temperature * constants["electric_constant"]
#         self.results["inveps"] = 1 - x_fluct / (const + M_fluct / self.volume)

#         return "FINISH PARA CONCLUDE"

#     def run(self, start=None, stop=None, step=None, n_proc=1):
#         parallel_exec(self._run, start, stop, step, n_proc)
