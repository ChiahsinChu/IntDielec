import csv
import glob
import os
import copy
import pickle
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from ..io.cp2k import Cp2kCube, Cp2kHartreeCube, Cp2kInput
from ..plot import core
from ..utils.math import *
from ..utils.unit import *


_EPSILON = VAC_PERMITTIVITY / UNIT_CHARGE * ANG_TO_M


class ElecEps:

    def __init__(
        self,
        atoms: Atoms = None,
        work_dir: str = None,
        v_zero: float = None,
        v_seq: list or np.ndarray = None,
        data_fmt: str = "json",
    ) -> None:
        self.atoms = atoms
        assert atoms.cell is not None

        self.work_dir = work_dir
        self.v_zero = v_zero
        self.set_v_seq(v_seq)
        self.data_fmt = data_fmt
        self._load_data()

        self.v_cubes = []
        self.e_cubes = []

    def ref_preset(self, calculate=False, **kwargs):
        """
        pp_dir="/data/jxzhu/basis", cutoff=400, eden=True
        """
        kwargs.update({
            "dip_cor": True,
            "hartree": True,
            "extended_fft_lengths": True,
        })

        dname = os.path.join(self.work_dir, "ref")
        if not os.path.exists(dname):
            os.makedirs(dname)

        task = Cp2kInput(self.atoms, **kwargs)
        task.write(output_dir=dname)

    def ref_calculate(self, vac_region):
        dname = os.path.join(self.work_dir, "ref")
        fname = glob.glob(os.path.join(dname, "*hartree*.cube"))
        assert len(fname) == 1
        cube = Cp2kHartreeCube(fname[0], vac_region)
        output = cube.get_ave_cube()
        self.set_v_zero(cube.potdrop)

    def preset(self, pos_dielec, calculate=False, **kwargs):
        kwargs.update({
            "dip_cor": False,
            "hartree": True,
            "eden": True,
            "extended_fft_lengths": True,
        })
        fp_params = {
            "GLOBAL": {
                "PROJECT": "cp2k"
            },
            "FORCE_EVAL": {
                "STRESS_TENSOR": "NONE",
                "DFT": {
                    "POISSON": {
                        "POISSON_SOLVER": "IMPLICIT",
                        "IMPLICIT": {
                            "BOUNDARY_CONDITIONS": "MIXED_PERIODIC",
                            "DIRICHLET_BC": {
                                "AA_PLANAR": [{
                                    "V_D":
                                    0.0,
                                    "PARALLEL_PLANE":
                                    "XY",
                                    "X_XTNT":
                                    "0.0 %f" % self.atoms.cell[0][0],
                                    "Y_XTNT":
                                    "0.0 %f" % self.atoms.cell[1][1],
                                    "INTERCEPT":
                                    pos_dielec[0],
                                    "PERIODIC_REGION":
                                    ".TRUE."
                                }, {
                                    "V_D":
                                    0.0,
                                    "PARALLEL_PLANE":
                                    "XY",
                                    "X_XTNT":
                                    "0.0 %f" % self.atoms.cell[0][0],
                                    "Y_XTNT":
                                    "0.0 %f" % self.atoms.cell[1][1],
                                    "INTERCEPT":
                                    pos_dielec[1],
                                    "PERIODIC_REGION":
                                    ".TRUE."
                                }]
                            }
                        }
                    }
                },
                "PRINT": {
                    "STRESS_TENSOR": {
                        "_": "OFF"
                    }
                }
            }
        }
        for v, task in zip(self.v_seq, self.v_tasks):
            dname = os.path.join(self.work_dir, task)
            if not os.path.exists(dname):
                os.makedirs(dname)

            task = Cp2kInput(self.atoms, **kwargs)
            fp_params["FORCE_EVAL"]["DFT"]["POISSON"]["IMPLICIT"][
                "DIRICHLET_BC"]["AA_PLANAR"][1]["V_D"] = self.v_zero + v
            task.write(output_dir=dname, fp_params=fp_params)

    def calculate(self, pos_vac, **kwargs):
        sigma = kwargs.get("gaussian_sigma", 0.0)

        efield_zero = []
        rho_e = []
        rho_e_conv = []
        for task in self.v_tasks:
            dname = os.path.join(self.work_dir, task)
            # hartree cube
            fname = glob.glob(os.path.join(dname, "*hartree*.cube"))
            assert len(fname) == 1
            cube = Cp2kHartreeCube(fname[0])
            output = cube.get_ave_cube(**kwargs)
            self.v_cubes.append(cube)
            efield_zero.append(self._calculate_efield_zero(cube, pos_vac))
            # eden cube
            fname = glob.glob(os.path.join(dname, "*ELECTRON_DENSITY*.cube"))
            assert len(fname) == 1
            cube = Cp2kCube(fname[0])
            output = cube.get_ave_cube(**kwargs)
            self.e_cubes.append(cube)
            rho_e.append(output[1])
            rho_e_conv.append(output[2])

            self.efield_zero = np.array(efield_zero)
            self.delta_efield_zero = np.diff(efield_zero, axis=0)
            self.rho_e = -np.array(rho_e)
            self.delta_rho_e = np.diff(self.rho_e, axis=0)
            self.rho_e_conv = -np.array(rho_e_conv)
            self.delta_rho_e_conv = np.diff(self.rho_e_conv, axis=0)

        self.results.update({0.0: {}})
        if sigma > 0.0:
            self.results.update({sigma: {}})
        for ii, v_prime in enumerate((self.v_seq[1:] + self.v_seq[:-1]) / 2):
            # print(v_prime)
            self.results[0.0].update({v_prime: {}})
            self._calculate_results(self.results[0.0][v_prime], output[0],
                                    self.delta_rho_e[ii],
                                    self.delta_efield_zero[ii])
            if sigma > 0.0:
                self.results[sigma].update({v_prime: {}})
                self._calculate_results(self.results[sigma][v_prime],
                                        output[0], self.delta_rho_e_conv[ii],
                                        self.delta_efield_zero[ii])

    def plot(self, sigma=0.0, fname=None):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[12, 8], dpi=300)
        xlabel = r"$z$ [A]"

        data_dict = self.results[sigma]
        v_primes = list(data_dict.keys())
        v_primes.sort()

        for ii, v_prime in enumerate(v_primes):
            # delta_rho
            axs[0][0].plot(data_dict[v_prime]["rho_pol"][0],
                           data_dict[v_prime]["rho_pol"][1],
                           color=plt.get_cmap('GnBu')(1 / (len(v_primes) + 1) *
                                                      (ii + 1)))
            # polarization
            axs[0][1].plot(data_dict[v_prime]["polarization"][0],
                           data_dict[v_prime]["polarization"][1],
                           color=plt.get_cmap('GnBu')(1 / (len(v_primes) + 1) *
                                                      (ii + 1)))
            # E-field
            axs[1][0].plot(data_dict[v_prime]["efield"][0],
                           data_dict[v_prime]["efield"][1],
                           color=plt.get_cmap('GnBu')(1 / (len(v_primes) + 1) *
                                                      (ii + 1)))
            # inveps
            axs[1][1].plot(data_dict[v_prime]["inveps"][0],
                           data_dict[v_prime]["inveps"][1],
                           color=plt.get_cmap('GnBu')(1 / (len(v_primes) + 1) *
                                                      (ii + 1)))

        ylabel = r"$\Delta \rho$ [a.u.]"
        axs[0][0].set_xlim(data_dict[v_prime]["rho_pol"][0].min(),
                           data_dict[v_prime]["rho_pol"][0].max())
        core.ax_setlabel(axs[0][0], xlabel, ylabel)
        axs[0][0].axhline(y=0.)

        ylabel = r"$\Delta P$ [a.u.]"
        axs[0][1].set_xlim(data_dict[v_prime]["polarization"][0].min(),
                           data_dict[v_prime]["polarization"][0].max())
        core.ax_setlabel(axs[0][1], xlabel, ylabel)
        axs[0][1].axhline(y=0.)

        ylabel = r"$\Delta E_z$ [V/A]"
        axs[1][0].set_xlim(data_dict[v_prime]["efield"][0].min(),
                           data_dict[v_prime]["efield"][0].max())
        core.ax_setlabel(axs[1][0], xlabel, ylabel)
        axs[1][0].axhline(y=0.)

        # ylabel = r"$\varepsilon_e^{-1}=\frac{\Delta E_z}{\Delta E_{z,vac}}$"
        ylabel = r"$\varepsilon_e^{-1}$"
        axs[1][1].set_xlim(data_dict[v_prime]["inveps"][0].min(),
                           data_dict[v_prime]["inveps"][0].max())
        core.ax_setlabel(axs[1][1], xlabel, ylabel)
        axs[1][1].axhline(y=0.)

        # color map
        cb_ax = fig.add_axes([.95, 0.15, .035, .7])
        cm = copy.copy(plt.get_cmap('GnBu'))
        norm = mpl.colors.Normalize(vmin=v_primes[0], vmax=v_primes[-1])
        im = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        fig.colorbar(im,
                     cax=cb_ax,
                     orientation='vertical',
                     ticks=np.linspace(v_primes[0], v_primes[-1], 5))

        fig.subplots_adjust(wspace=0.25, hspace=0.25)

        if fname:
            fig.savefig(fname, bbox_inches='tight')

        return fig, axs

    def workflow(self):
        # connect all together
        # hang in the background and check the output file (finished_tag)
        pass

    def set_v_zero(self, v_zero: float = None):
        self.v_zero = v_zero

    def set_v_seq(self, v_seq: list or np.ndarray = None):
        self.v_seq = v_seq
        self.v_tasks = []
        if v_seq is not None:
            for v in v_seq:
                if v >= 0:
                    self.v_tasks.append("%.1f" % v)
                else:
                    self.v_tasks.append("_%.1f" % -v)

    def _load_data(self):
        fname = os.path.join(self.work_dir, "eps_data.%s" % self.data_fmt)
        if os.path.exists(fname):
            getattr(self, "_load_data_%s" % self.data_fmt)(fname)
        else:
            self.results = {}

    def _load_data_csv(self, fname):
        with open(fname, "r") as f:
            data = csv.reader(f)
            self.results = {rows[0]: rows[1] for rows in data}

    def _load_data_json(self, fname):
        with open(fname, "r") as f:
            self.results = json.load(f)

    def _load_data_pkl(self, fname):
        with open(fname, "rb") as f:
            self.results = pickle.load(f)

    def _save_data(self):
        fname = os.path.join(self.work_dir, "eps_data.%s" % self.data_fmt)
        getattr(self, "_save_data_%s" % self.data_fmt)(fname)

    def _save_data_csv(self, fname):
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results.keys())
            writer.writeheader()
            writer.writerow(self.results)

    def _save_data_json(self, fname):
        with open(fname, "w") as f:
            json.dump(self.results, f)

    def _save_data_pkl(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self.results, f)

    @staticmethod
    def _calculate_efield_zero(cube, pos_vac):
        # E-field [V/A]
        x, y = get_dev(cube.ave_grid, cube.ave_cube_data)
        vac_id = np.argmin(np.abs(x - pos_vac))
        return y[vac_id]

    @staticmethod
    def _calculate_polarization(x, delta_rho_e):
        x, _y = get_int(x, delta_rho_e)
        # transfer the Angstrom in integration to Bohr
        y = -_y / AU_TO_ANG
        return x, y

    @staticmethod
    def _calculate_efield(x, delta_rho_e, delta_efield_zero):
        x, _y = get_int(x, delta_rho_e)
        # E-field [V/A]
        y = _y / (AU_TO_ANG)**3 / _EPSILON
        return x, (y + delta_efield_zero)

    @staticmethod
    def _calculate_inveps(x, delta_rho_e, delta_efield_zero):
        x, _y = get_int(x, delta_rho_e)
        # E-field [V/A]
        y = _y / (AU_TO_ANG)**3 / _EPSILON
        return x, (y + delta_efield_zero) / delta_efield_zero

    def _calculate_results(self, out_dict, x_in, delta_rho_e,
                           delta_efield_zero):
        # delta_rho
        out_dict["rho_pol"] = (x_in, delta_rho_e)
        # E-field
        x, y = self._calculate_efield(x_in, delta_rho_e, delta_efield_zero)
        out_dict["efield"] = (x, y)
        # polarization
        x, y = self._calculate_polarization(x_in, delta_rho_e)
        out_dict["polarization"] = (x, y)
        # inveps
        x, y = self._calculate_inveps(x_in, delta_rho_e, delta_efield_zero)
        out_dict["inveps"] = (x, y)