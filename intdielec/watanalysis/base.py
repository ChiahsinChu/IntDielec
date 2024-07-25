import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from scipy import constants

from ..utils.mda import make_selection


class SelectedTemperature(AnalysisBase):
    """
    TBC
    
    CP2K velocity: https://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/PRINT/VELOCITIES.html
    """
    def __init__(self,
                 ag,
                 u_vels=None,
                 zero_p=False,
                 v_format="xyz",
                 unit="au",
                 verbose=False):
        self.ag = ag
        trajectory = ag.universe.trajectory
        super().__init__(trajectory, verbose=verbose)
        self.n_frames = len(trajectory)
        if u_vels is not None:
            self.vels = u_vels.trajectory
        else:
            self.vels = None
        self.zero_p = zero_p
        self.v_fmt = v_format
        self.unit = unit

    def _prepare(self):
        self.temperature = np.zeros((self.n_frames), dtype=np.float64)

    def _single_frame(self):
        # u.trajectory[ii].positions
        if self.vels is None:
            ts_vels = self._ts.velocities
        else:
            ts_vels = self.vels[self._frame_index].positions
        ts_vels2 = np.sum(ts_vels * ts_vels, axis=-1)
        ts_vels2 = ts_vels2[self.ag.indices]
        ts_masses = self.ag.masses
        if self.zero_p:
            ts_dgf = 3 * len(self.ag) - 3
        else:
            ts_dgf = 3 * len(self.ag)
        self.temperature[self._frame_index] = np.sum(
            ts_vels2 * ts_masses) / ts_dgf

    def _conclude(self):
        if self.unit == "au":
            prefactor = constants.atomic_mass * (
                constants.physical_constants["Bohr radius"][0] /
                constants.physical_constants["atomic unit of time"][0]
            )**2 / constants.Boltzmann
            # print("au prefactor: ", prefactor)
        elif self.unit == "metal":
            prefactor = constants.atomic_mass * (
                constants.angstrom / constants.pico)**2 / constants.Boltzmann
        else:
            raise AttributeError("Unsupported unit %s" % self.unit)
        self.temperature *= prefactor
        return self.temperature


class InterfaceTemperature(SelectedTemperature):
    def __init__(self,
                 universe,
                 u_vels=None,
                 zero_p=False,
                 v_format="xyz",
                 unit="au",
                 verbose=False,
                 **kwargs):
        select = make_selection(**kwargs)
        # print("selection: ", select)
        super().__init__(universe.select_atoms(select, updating=True), u_vels,
                         zero_p, v_format, unit, verbose)

class ReWeightingOP:
    def __init__(
        self,
        temp_in,
        temp_out,
        energy_in,
        energy_out,
    ) -> None:
        self.temp_in = temp_in
        self.temp_out = temp_out
        self.energy_in = np.array(energy_in) - np.mean(energy_in)
        self.energy_out = np.array(energy_out) - np.mean(energy_out)
        
    def calc_weights(self):
        beta_in = 1. / (self.temp_in * constants.physical_constants["Boltzmann constant in eV/K"][0])
        beta_out = 1. / (self.temp_out * constants.physical_constants["Boltzmann constant in eV/K"][0])
        exp_values = beta_in * self.energy_in - beta_out * self.energy_out
        weights = np.exp(exp_values)
        return weights