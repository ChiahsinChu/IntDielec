import copy
import glob
import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, io
from scipy import optimize, stats

from .. import plot
from ..calculator.cp2k import Cp2kCalculator
from ..io.cp2k import (Cp2kCube, Cp2kHartreeCube, Cp2kInput, Cp2kOutput,
                       Cp2kPdos)
from ..utils.config import check_water
from ..utils.math import *
from ..utils.unit import *
from ..utils.utils import load_dict, save_dict, update_dict
from . import Eps

_EPSILON = VAC_PERMITTIVITY / UNIT_CHARGE * ANG_TO_M
N_SURF = 16
L_WAT = 15.
L_VAC = 10.
EPS_WAT = 2.
L_WAT_PDOS = 10.
MAX_LOOP = 10
SEARCH_CONVERGENCE = 1e-1
V_GUESS_BOUND = [-(L_WAT + EPS_WAT * L_VAC * 2), L_WAT + EPS_WAT * L_VAC * 2]

plot.use_style("pub")


class ElecEps(Eps):
    def __init__(
        self,
        atoms: Atoms = None,
        work_dir: str = None,
        v_zero: float = None,
        v_ref: float = 0.,
        v_seq: list or np.ndarray = None,
        data_fmt: str = "pkl",
    ) -> None:
        super().__init__(work_dir, data_fmt)

        if atoms is None:
            atoms = io.read(os.path.join(work_dir, "ref/coord.xyz"))
        self.atoms = atoms
        assert atoms.cell is not None

        self.v_zero = v_zero
        self.v_ref = v_ref
        self.set_v_seq(v_seq)

        self.v_cubes = []
        self.e_cubes = []

        self._setup()

        logging.info("Number of atoms: %d" % len(atoms))

    def ref_preset(self, fp_params={}, dname="ref", calculate=False, **kwargs):
        """
        pp_dir="/data/jxzhu/basis", cutoff=400, eden=True
        """
        update_d = {
            "dip_cor": True,
            "hartree": True,
            "extended_fft_lengths": True,
        }
        update_dict(kwargs, update_d)

        update_dict(fp_params,
                    self._water_pdos_input(n_wat=self.info["n_wat"]))

        dname = os.path.join(self.work_dir, dname)
        if not os.path.exists(dname):
            os.makedirs(dname)

        task = Cp2kInput(self.atoms, **kwargs)
        task.write(output_dir=dname, fp_params=fp_params, save_dict=calculate)

    def ref_calculate(self, vac_region=None, dname="ref"):
        dname = os.path.join(self.work_dir, dname)
        try:
            fname = glob.glob(os.path.join(dname, "output*"))
            assert len(fname) == 1
            output = Cp2kOutput(fname[0])
            DeltaV = output.potdrop[0]
        except:
            assert (vac_region is not None)
            fname = glob.glob(os.path.join(dname, "*hartree*.cube"))
            assert len(fname) == 1
            cube = Cp2kHartreeCube(fname[0], vac_region)
            output = cube.get_ave_cube()
            DeltaV = cube.potdrop
        self.set_v_zero(DeltaV)

    def preset(self, pos_dielec, fp_params={}, calculate=False, **kwargs):
        wfn_restart = os.path.join(self.work_dir, "ref", "cp2k-RESTART.wfn")
        update_d = {
            "dip_cor": False,
            "hartree": True,
            "eden": True,
            "totden": True,
            "extended_fft_lengths": True,
            "wfn_restart": wfn_restart
        }
        update_dict(kwargs, update_d)

        update_d = {
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
                                    self.v_ref,
                                    "PARALLEL_PLANE":
                                    "XY",
                                    "X_XTNT":
                                    "0.0 %.4f" % self.atoms.cell[0][0],
                                    "Y_XTNT":
                                    "0.0 %.4f" % self.atoms.cell[1][1],
                                    "INTERCEPT":
                                    pos_dielec[0],
                                    "PERIODIC_REGION":
                                    ".TRUE."
                                }, {
                                    "V_D":
                                    self.v_ref,
                                    "PARALLEL_PLANE":
                                    "XY",
                                    "X_XTNT":
                                    "0.0 %.4f" % self.atoms.cell[0][0],
                                    "Y_XTNT":
                                    "0.0 %.4f" % self.atoms.cell[1][1],
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
        update_dict(fp_params, update_d)
        update_dict(fp_params,
                    self._water_pdos_input(n_wat=self.info["n_wat"]))

        for v, task in zip(self.v_seq, self.v_tasks):
            dname = os.path.join(self.work_dir, task)
            if not os.path.exists(dname):
                os.makedirs(dname)

            task = Cp2kInput(self.atoms, **kwargs)
            fp_params["FORCE_EVAL"]["DFT"]["POISSON"]["IMPLICIT"][
                "DIRICHLET_BC"]["AA_PLANAR"][1][
                    "V_D"] = self.v_ref + self.v_zero + v
            task.write(output_dir=dname,
                       fp_params=fp_params,
                       save_dict=calculate)

    def calculate(self, pos_vac, save_fname="eps_data", **kwargs):
        """
        If v does not exist or overwrite is True, then read the data
        - sigma 
            - [x] v
            - [x] hartree
            - [x] rho
            - [x] efield_vac
            - [ ] mo

            - [x] v_prime
            - [x] rho_pol
            - [x] efield
            - [x] polarization
            - [x] inveps
            - [x] std_inveps
        """
        sigma = kwargs.get("gaussian_sigma", 0.0)
        update_dict(self.results, {0.0: {}})
        old_v = self.results[0.0].get("v", [])
        if not isinstance(old_v, list):
            old_v = old_v.tolist()
        efield = self.results[0.0].get("efield", [])
        if not isinstance(efield, list):
            efield = efield.tolist()
        hartree = self.results[0.0].get("hartree", [])
        if not isinstance(hartree, list):
            hartree = hartree.tolist()
        efield_vac = self.results[0.0].get("efield_vac", [])
        if not isinstance(efield_vac, list):
            efield_vac = efield_vac.tolist()
        rho = self.results[0.0].get("rho", [])
        if not isinstance(rho, list):
            rho = rho.tolist()
        # mo = self.results[0.0].get("mo", [])

        if sigma > 0:
            update_dict(self.results, {sigma: {}})
            old_v_conv = self.results[sigma].get("v", [])
            rho_conv = self.results[sigma].get("rho", [])

        calculate_delta_flag = False
        for v, task in zip(self.v_seq, self.v_tasks):
            if not False in (np.abs(v - np.array(old_v)) > 1e-2):
                calculate_delta_flag = True
                old_v.append(v)
                efield.append(v / self.atoms.cell[2][2])

                dname = os.path.join(self.work_dir, task)
                # hartree cube
                fname = glob.glob(os.path.join(dname, "*hartree*.cube"))
                assert len(fname) == 1
                cube = Cp2kHartreeCube(fname[0])
                output = cube.get_ave_cube(**kwargs)
                hartree.append(output[1])
                # efield in the vac
                efield_vac.append(
                    self._calculate_efield_zero(output[0], output[1], pos_vac))
                # eden cube
                try:
                    fname = glob.glob(
                        os.path.join(dname, "*TOTAL_DENSITY*.cube"))
                    assert len(fname) == 1
                    cube = Cp2kCube(fname[0])
                except:
                    fname = glob.glob(
                        os.path.join(dname, "*ELECTRON_DENSITY*.cube"))
                    assert len(fname) == 1
                    cube = Cp2kCube(fname[0])
                output = cube.get_ave_cube(**kwargs)
                rho.append(-output[1])
                if sigma > 0:
                    old_v_conv.append(v)
                    rho_conv.append(-output[2])
                    # self.rho_conv = -np.array(rho_conv)
                # TODO: water PDOS

        if calculate_delta_flag:
            # update data
            sort_ids = np.argsort(old_v)
            self.results[0.0]["v"] = np.sort(old_v)
            self.results[0.0]["efield"] = np.array(efield)[sort_ids]
            self.results[0.0]["v_grid"] = output[0]
            self.results[0.0]["hartree"] = np.array(hartree)[sort_ids]
            self.results[0.0]["efield_vac"] = np.array(efield_vac)[sort_ids]
            self.results[0.0]["rho"] = np.array(rho)[sort_ids]
            # self.results[0.0]["mo"] = np.array(mo)[sort_ids]
            self.calculate_delta(0.0)
            if sigma > 0:
                sort_ids = np.argsort(old_v_conv)
                self.results[sigma]["v"] = np.sort(old_v_conv)
                self.results[sigma]["rho"] = np.array(rho_conv)[sort_ids]
                self.results[sigma]["efield_vac"] = np.array(
                    efield_vac)[sort_ids]
                self.calculate_delta(sigma)

        self._save_data(save_fname)

    def calculate_delta(self, sigma):
        data_dict = self.results[sigma]
        data_dict["v_prime"] = (np.array(data_dict["v"])[1:] +
                                np.array(data_dict["v"])[:-1]) / 2
        data_dict["rho_pol"] = np.diff(data_dict["rho"], axis=0)
        data_dict["delta_efield_vac"] = np.diff(data_dict["efield_vac"],
                                                axis=0)
        x, y = self._calculate_efield(data_dict["v_grid"],
                                      data_dict["rho_pol"],
                                      data_dict["delta_efield_vac"])
        data_dict["delta_efield"] = y
        data_dict["v_prime_grid"] = x
        data_dict["delta_pol"] = self._calculate_polarization(
            data_dict["v_grid"], data_dict["rho_pol"])
        data_dict["inveps"] = self._calculate_inveps(
            data_dict["v_grid"], data_dict["rho_pol"],
            data_dict["delta_efield_vac"])
        data_dict["lin_test"] = np.std(data_dict["inveps"], axis=0)

    def make_plots(self, out=None, sigma=0.0):
        if not os.path.exists(os.path.join(self.work_dir, "figures")):
            os.makedirs(os.path.join(self.work_dir, "figures"))
        if out is None:
            out = [["hartree", "rho_pol"], ["delta_efield", "inveps"]]

        fnames = glob.glob(os.path.join(self.work_dir, "*.%s" % self.data_fmt))
        for data_fname in fnames:
            if not "task_info" in data_fname:
                # read data
                data_dict = load_dict(data_fname)[sigma]

                fig, axs = self._make_plots(out,
                                            data_dict,
                                            scale=(np.min(data_dict["efield"]),
                                                   np.max(
                                                       data_dict["efield"])))
                figure_name = os.path.splitext(os.path.basename(data_fname))[0]
                fig.savefig(os.path.join(self.work_dir, "figures",
                                         "%s.png" % figure_name),
                            bbox_inches='tight')

    def _make_plots(self, out, data_dict, scale):
        """
        - Hartree
        - rho_pol
        - E-field
        - polarization
        - inveps
        - lin_test
        """

        ylabels_dict = {
            "hartree": r"$V_H$ [Hartree]",
            "rho_pol": r"$\rho_{pol}$ [e/bohr$^3$]",
            "delta_efield": r"$\Delta E_z$ [V/A]",
            "delta_pol": r"$\Delta P$ [e/bohr$^2$]",
            "inveps": r"$\varepsilon_e^{-1}$",
            "lin_test": r"$\sigma(\varepsilon_e^{-1})$"
        }
        v_primes = list(data_dict.keys())
        v_primes.sort()
        # TODO: add case for one row/col!
        shape = np.shape(out)
        nrows = shape[0]
        ncols = shape[1]
        fig, axs = plt.subplots(nrows=nrows,
                                ncols=ncols,
                                figsize=[ncols * 5, nrows * 3],
                                dpi=300)
        xlabel = r"$z$ [A]"

        for ii in range(nrows):
            for jj in range(ncols):
                ax = axs[ii][jj]
                kw = out[ii][jj]
                ylabel = ylabels_dict[kw]

                ys = data_dict[kw]
                shape = np.shape(ys)
                if len(data_dict["v"]) == shape[0]:
                    labels = data_dict["efield"]
                else:
                    labels = np.diff(data_dict["efield"], axis=0)
                if len(data_dict["v_grid"]) == shape[1]:
                    xs = np.tile(data_dict["v_grid"], (shape[1], 1))
                else:
                    xs = np.tile(data_dict["v_prime_grid"], (shape[1], 1))

                plot.ax_colormap_lines(ax,
                                       xs,
                                       ys,
                                       labels=labels,
                                       scale=scale,
                                       colormap="coolwarm")
                plot.ax_setlabel(ax, xlabel, ylabel)

                ax.axhline(y=0., color="gray")
                ax.axvline(x=L_VAC, color="gray")
                ax.axvline(x=np.max(xs[0]) - L_VAC, color="gray")
                ax.axvline(x=np.max(xs[0]) - L_VAC - L_WAT, color="gray")
                ax.set_xlim(np.min(xs[0]), np.max(xs[0]))

        # color map
        cb_ax = fig.add_axes([.95, 0.15, .035, .7])
        cm = copy.copy(plt.get_cmap("coolwarm"))
        norm = mpl.colors.Normalize(vmin=scale[0], vmax=scale[1])
        im = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        fig.colorbar(im,
                     cax=cb_ax,
                     orientation='vertical',
                     ticks=np.linspace(scale[0], scale[1], 5))
        cb_ax.set_title("E-field [V/A]", fontsize="medium", y=1.1)

        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        return fig, axs

    def plot(self, sigma=0.0, fname="eps_cal.png"):
        """
        deprecated
        """
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

        ylabel = r"$\rho_{pol}$ [a.u.]"
        axs[0][0].set_xlim(data_dict[v_prime]["rho_pol"][0].min(),
                           data_dict[v_prime]["rho_pol"][0].max())
        plot.ax_setlabel(axs[0][0], xlabel, ylabel)
        axs[0][0].axhline(y=0., ls="--", color="gray")

        ylabel = r"$\Delta P$ [a.u.]"
        axs[0][1].set_xlim(data_dict[v_prime]["polarization"][0].min(),
                           data_dict[v_prime]["polarization"][0].max())
        plot.ax_setlabel(axs[0][1], xlabel, ylabel)
        axs[0][1].axhline(y=0., ls="--", color="gray")

        ylabel = r"$\Delta E_z$ [V/A]"
        axs[1][0].set_xlim(data_dict[v_prime]["efield"][0].min(),
                           data_dict[v_prime]["efield"][0].max())
        plot.ax_setlabel(axs[1][0], xlabel, ylabel)
        axs[1][0].axhline(y=0., ls="--", color="gray")

        # ylabel = r"$\varepsilon_e^{-1}=\frac{\Delta E_z}{\Delta E_{z,vac}}$"
        ylabel = r"$\varepsilon_e^{-1}$"
        axs[1][1].set_xlim(data_dict[v_prime]["inveps"][0].min(),
                           data_dict[v_prime]["inveps"][0].max())
        plot.ax_setlabel(axs[1][1], xlabel, ylabel)
        axs[1][1].axhline(y=0., ls="--", color="gray")
        axs[1][1].axhline(y=1., ls="--", color="gray")

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
            fig.savefig(os.path.join(self.work_dir, fname),
                        bbox_inches='tight')

        return fig, axs

    def lin_test(self, sigma=0.0, fname="eps_lin_test.png"):
        """
        deprecated
        """
        data = self.results[sigma]
        inveps_data = []
        for v in data.values():
            inveps_data.append(v["inveps"][1].tolist())
        std_inveps_data = np.std(inveps_data, axis=0)

        fig, ax = plt.subplots(figsize=[6, 4], dpi=300)
        xlabel = r"$z$ [A]"
        ylabel = r"$\sigma(\varepsilon_e^{-1})$"

        ax.plot(v["inveps"][0], std_inveps_data, color="black")

        ax.axhline(y=std_inveps_data[-1], color="gray", ls="--")
        ax.set_xlim(v["inveps"][0].min(), v["inveps"][0].max())
        plot.ax_setlabel(ax, xlabel, ylabel)

        if fname:
            fig.savefig(os.path.join(self.work_dir, fname),
                        bbox_inches='tight')

        return fig, ax

    def workflow(self,
                 configs: str = "param.json",
                 ignore_finished_tag: bool = False):
        """
        Example for `param.json`:
        ```json
        {
            "load_module": [
                "intel/17.5.239", 
                "mpi/intel/2017.5.239", 
                "gcc/5.5.0", 
                "cp2k/7.1"
            ],
            "command": "mpiexec.hydra cp2k_shell.popt",
            "ref_preset": {},
            "ref_calculate": {},
            "preset": {},
            "calculate": {}
        }
        ```
        """
        default_command = "mpiexec.hydra cp2k.popt"
        super().workflow(configs, default_command)

        # ref: preset
        logging.info(
            "{:=^50}".format(" Start: set up files for dipole correction "))
        tmp_params = self.wf_configs.get("ref_preset", {})
        self.ref_preset(calculate=True, **tmp_params)
        logging.info(
            "{:=^50}".format(" End: set up files for dipole correction "))

        # ref: DFT calculation
        self._dft_calculate(os.path.join(self.work_dir, "ref"),
                            ignore_finished_tag)

        # ref: calculate dipole moment
        logging.info(
            "{:=^50}".format(" Start: analyse dipole correction data "))
        tmp_params = self.wf_configs.get("ref_calculate", {})
        self.ref_calculate(**tmp_params)
        logging.info("{:=^50}".format(" End: analyse dipole correction data "))

        # eps_cal: preset
        logging.info(
            "{:=^50}".format(" Start: set up files for Efield calculation "))
        tmp_params = self.wf_configs.get("preset", {})
        self.preset(calculate=True, **tmp_params)
        logging.info(
            "{:=^50}".format(" End: set up files for Efield calculation "))
        # eps_cal: DFT calculation
        for task in self.v_tasks:
            self._dft_calculate(os.path.join(self.work_dir, task),
                                ignore_finished_tag)

        # eps_cal: calculate eps
        logging.info("{:=^50}".format(" Start: analyse eps calculation "))
        tmp_params = self.wf_configs.get("calculate", {})
        self.calculate(**tmp_params)
        logging.info("{:=^50}".format(" End: analyse eps calculation "))

        logging.info("{:=^50}".format(" End: eps Calculation "))

    def set_v_zero(self, v_zero: float):
        self.v_zero = v_zero

    def set_v_ref(self, v_ref: float):
        self.v_ref = v_ref

    def set_v_seq(self, v_seq: list or np.ndarray = None):
        self.v_seq = v_seq
        self.v_tasks = []
        if v_seq is not None:
            for v in v_seq:
                if v >= 0:
                    self.v_tasks.append("%.1f" % v)
                else:
                    self.v_tasks.append("_%.1f" % -v)

    @staticmethod
    def _calculate_efield_zero(x_ori, y_ori, pos_vac):
        # E-field [V/A]
        x, y = get_dev(x_ori, y_ori)
        vac_id = np.argmin(np.abs(x - pos_vac))
        return y[vac_id]

    @staticmethod
    def _calculate_efield(x, delta_rho_e, delta_efield_zero):
        x, _y = get_int_array(x, delta_rho_e)
        # E-field [V/A]
        y = _y / (AU_TO_ANG)**3 / _EPSILON
        return x, (y + delta_efield_zero.reshape(-1, 1))

    @staticmethod
    def _calculate_polarization(x, delta_rho_e):
        x, _y = get_int_array(x, delta_rho_e)
        # transfer the Angstrom in integration to Bohr
        y = -_y / AU_TO_ANG
        return y

    @staticmethod
    def _calculate_inveps(x, delta_rho_e, delta_efield_zero):
        x, _y = get_int_array(x, delta_rho_e)
        # E-field [V/A]
        y = _y / (AU_TO_ANG)**3 / _EPSILON
        return (y + delta_efield_zero.reshape(
            -1, 1)) / delta_efield_zero.reshape(-1, 1)

    # def _calculate_results(self, out_dict, x_in, delta_rho_e,
    #                        delta_efield_zero):
    #     # TODO: hartree
    #     out_dict["hartree"] = (x_in, delta_rho_e)
    #     # delta_rho
    #     out_dict["rho_pol"] = (x_in, delta_rho_e)
    #     # E-field
    #     x, y = self._calculate_efield(x_in, delta_rho_e, delta_efield_zero)
    #     out_dict["efield"] = (x, y)
    #     # polarization
    #     x, y = self._calculate_polarization(x_in, delta_rho_e)
    #     out_dict["polarization"] = (x, y)
    #     # inveps
    #     x, y = self._calculate_inveps(x_in, delta_rho_e, delta_efield_zero)
    #     out_dict["inveps"] = (x, y)
    #     # TODO: lin_test

    #     # TODO: water pdos

    def _dft_calculate(self, work_dir, ignore_finished_tag):
        flag = os.path.join(work_dir, "finished_tag")
        if (ignore_finished_tag == True) or (os.path.exists(flag) == False):
            Cp2kCalculator(work_dir=work_dir).run(type="bash",
                                                  command=self.command)

    @staticmethod
    def _water_pdos_input(n_wat):
        update_d = {"FORCE_EVAL": {"DFT": {"PRINT": {"PDOS": {"LDOS": []}}}}}
        for ii in range(n_wat):
            id_start = ii * 3 + 1
            id_end = (ii + 1) * 3
            update_d["FORCE_EVAL"]["DFT"]["PRINT"]["PDOS"]["LDOS"].append(
                {"LIST": "%d..%d" % (id_start, id_end)})
        return update_d

    def _setup(self):
        self.info = {}

        info_dict = self.info
        atoms = self.atoms

        info_dict["atype"] = np.array(atoms.get_chemical_symbols())

        info_dict["O_mask"] = (info_dict["atype"] == "O")
        info_dict["H_mask"] = (info_dict["atype"] == "H")
        info_dict["water_mask"] = info_dict["O_mask"] + info_dict["H_mask"]
        info_dict["metal_mask"] = ~info_dict["water_mask"]
        info_dict["n_wat"] = len(
            info_dict["atype"][info_dict["water_mask"]]) // 3

        logging.info("Number of water molecules: %d" % info_dict["n_wat"])

        z = self.ref_lo_atoms.get_positions()[info_dict["metal_mask"], 2]
        info_dict["z_ave"] = np.sort(z)[-N_SURF:].mean()
        logging.info("Position of metal surface: %.3f [A]" %
                     info_dict["z_ave"])


class IterElecEps(ElecEps):
    def __init__(self,
                 atoms: Atoms = None,
                 work_dir: str = None,
                 data_fmt: str = "pkl") -> None:
        Eps.__init__(self, work_dir, data_fmt)

        if atoms is None:
            atoms = io.read(os.path.join(work_dir, "pbc/coord.xyz"))
        self.atoms = atoms
        assert atoms.cell is not None

        self._setup("pbc")

        self.v_ref = 0.
        self.v_cubes = []
        self.e_cubes = []

    def pbc_preset(self, fp_params={}, dname="pbc", calculate=False, **kwargs):
        kwargs.update({"dip_cor": False})

        n_wat = self.pbc_info["n_wat"]
        update_d = self._water_pdos_input(n_wat=n_wat)
        update_dict(fp_params, update_d)

        dname = os.path.join(self.work_dir, dname)
        if not os.path.exists(dname):
            os.makedirs(dname)

        task = Cp2kInput(self.pbc_atoms, **kwargs)
        task.write(output_dir=dname, fp_params=fp_params, save_dict=calculate)

        self.work_subdir = dname

    def pbc_calculate(self):
        n_wat = self.pbc_info["n_wat"]
        z_wat = self.pbc_atoms.get_positions()[self.pbc_info["O_mask"], 2]
        z_wat_vs_lo = z_wat - self.pbc_info["z_lo"]
        z_wat_vs_hi = self.pbc_info["z_hi"] - z_wat

        sort_ids = np.argsort(z_wat)

        cbm, vbm = self._water_mo_output(n_wat)

        np.save(os.path.join(self.work_subdir, "data.npy"), [
            z_wat_vs_lo[sort_ids], z_wat_vs_hi[sort_ids], cbm[sort_ids],
            vbm[sort_ids]
        ])

    def ref_preset(self, fp_params={}, calculate=False, **kwargs):
        # lower surface
        self.atoms = self._convert()
        self._setup("ref_lo")
        update_dict(fp_params,
                    self._water_pdos_input(n_wat=self.ref_lo_info["n_wat"]))
        super().ref_preset(fp_params=fp_params,
                           dname="ref_lo",
                           calculate=calculate,
                           **kwargs)
        # upper surface
        self.atoms = self._convert(inverse=True)
        self._setup("ref_hi")
        update_dict(fp_params,
                    self._water_pdos_input(n_wat=self.ref_hi_info["n_wat"]))
        super().ref_preset(fp_params=fp_params,
                           dname="ref_hi",
                           calculate=calculate,
                           **kwargs)

    def ref_calculate(self, vac_region=None):
        dname = "ref_%s" % self.suffix
        self.info_dict = getattr(self, "%s_info" % dname)
        self.work_subdir = os.path.join(self.work_dir, dname)
        self.atoms = getattr(self, "%s_atoms" % dname)
        super().ref_calculate(vac_region=vac_region, dname=dname)
        # setattr(IterElecEps, "v_zero_%s" % suffix, self.v_zero)
        self.info_dict["v_zero"] = self.v_zero
        logging.debug("V_zero: %f" % self.v_zero)
        if not os.path.exists(os.path.join(self.work_subdir, "data.npy")):
            n_wat = self.info_dict["n_wat"]
            z_wat = self.atoms.get_positions()[self.info_dict["O_mask"],
                                               2] - self.info_dict["z_ave"]
            sort_ids = np.argsort(z_wat)
            cbm, vbm = self._water_mo_output(n_wat)

            np.save(os.path.join(self.work_subdir, "data.npy"),
                    [z_wat[sort_ids], cbm[sort_ids], vbm[sort_ids]])
        self.v_seq = [self._guess()]

    def _guess(self, type="linear", **kwargs):
        logging.info("V_guess [V]: %f" % self.v_guess)
        # logging.debug("working directory: %s" % self.work_subdir)
        ref_data = np.load(os.path.join(self.work_dir, "pbc/data.npy"))
        test_data = np.load(os.path.join(self.work_subdir, "data.npy"))
        ref_id = np.argmin(np.abs(test_data[0] - L_WAT_PDOS))
        self.convergence = test_data[-1][:ref_id].mean()
        if self.suffix == "lo":
            self.convergence -= ref_data[-1][:ref_id].mean()
        else:
            self.convergence -= ref_data[-1][-ref_id:].mean()
        logging.info("Convergence [V]: %f" % self.convergence)
        try:
            self.search_history = np.append(
                self.search_history,
                np.array([[self.v_guess, self.convergence]]),
                axis=0)
        except:
            self.search_history = np.append(
                self.search_history, np.array([self.v_guess,
                                               self.convergence]))
            self.search_history = self.search_history.reshape(-1, 2)
        self.v_guess = getattr(self, "_guess_%s" % type)(**kwargs)
        return self.v_guess

    def _guess_simple(self):
        z = self.atoms.get_positions()[:, 2]
        mask_coord = (z >= (self.info_dict["z_ave"] + L_WAT_PDOS))
        mask = self.info_dict["O_mask"] * mask_coord
        sel_water_ids = np.arange(len(self.atoms))[mask] // 3
        n_e = 0.
        for ii in sel_water_ids:
            # logging.debug("selected water index: %d" % ii)
            fname = os.path.join(self.work_subdir,
                                 "cp2k-list%d-1.pdos" % (ii + 1))
            pdos = Cp2kPdos(fname)
            e = pdos.energies - pdos.fermi
            mask = ((e - 0.1) * (e + 0.1) < 0.)
            raw_dos = pdos._get_raw_dos("total")
            occupation = pdos.occupation
            n_e += (raw_dos[mask] * (2.0 - occupation[mask])).sum()
        logging.debug("number of electron at wat/vac interface: %f" % n_e)

        cross_area = np.linalg.norm(
            np.cross(self.atoms.cell[0], self.atoms.cell[1]))
        logging.debug("cross area: %f" % cross_area)
        v_guess = 2 * L_VAC * (n_e / cross_area / _EPSILON)
        logging.debug("V_guess (1): %f" % v_guess)

        # dielectrics
        # slope = (EPS_WAT * L_VAC * 2 + L_WAT) / (EPS_WAT * L_VAC + L_WAT_PDOS)
        # emprical values
        slope = 0.5
        ref_data = np.load(os.path.join(self.work_dir, "pbc/data.npy"))
        test_data = np.load(os.path.join(self.work_subdir, "data.npy"))
        ref_id = np.argmin(np.abs(test_data[0] - L_WAT_PDOS))
        logging.debug("ref_id: %d" % ref_id)
        test_homo = test_data[-1][(ref_id - 4):(ref_id + 1)].mean()
        if self.suffix == "lo":
            ref_homo = ref_data[-1][(ref_id - 4):(ref_id + 1)].mean()
        else:
            ref_homo = ref_data[-1][-(ref_id - 1):-(ref_id - 6)].mean()
        delta_v = test_homo - ref_homo
        v_guess += delta_v * slope
        logging.debug("V_guess (2): %f" % v_guess)
        return v_guess

    def _guess_optimize(self, n_step=2):
        if len(self.search_history) < n_step:
            return self._guess_simple()
        else:

            def func(x):
                y = np.interp([x],
                              xp=self.search_history[:, 0],
                              fp=self.search_history[:, 1])
                return y[0]

            id_argmin = np.argmin(np.abs(self.search_history[:, 1]))
            x0 = self.search_history[:, 0][id_argmin]
            v_guess = optimize.fsolve(func=func,
                                      x0=x0,
                                      xtol=SEARCH_CONVERGENCE)[0]

            # avoid trapping
            if np.abs(x0 - v_guess) < 1e-3:
                coeff = np.random.uniform() * 0.1 + 0.1
                v_guess += (coeff * self.search_history[:, 1][id_argmin] /
                            np.abs(self.search_history[:, 1][id_argmin]))
            # avoid the guess goes mad...
            v_guess = min(max(v_guess, V_GUESS_BOUND[0]), V_GUESS_BOUND[1])
            return v_guess

    def _guess_linear(self):
        if len(self.search_history) < 2:
            return self._guess_simple()
        else:
            # dataset = self.search_history[np.argsort(self.search_history[:,
            #                                                              0])]
            # id_argmin = np.argmin(np.abs(dataset[:, 1]))
            # if id_argmin == 0:
            #     # left endpoint
            #     data = dataset[:id_argmin + 2]
            # elif id_argmin == (len(dataset) - 1):
            #     # right endpoint
            #     data = dataset[id_argmin - 1:]
            # else:
            #     data = dataset[id_argmin - 1:id_argmin + 2]
            result = stats.linregress(x=self.search_history[:, 0],
                                      y=self.search_history[:, 1],
                                      alternative="less")
            if np.abs(result.slope) < 0.1:
                v_guess = self.search_history[-1, 0] + 5.
            else:
                v_guess = -result.intercept / max(result.slope, -10.)
        return v_guess

    def search_preset(self, n_iter, fp_params={}, calculate=False, **kwargs):
        dname = "search_%s.%06d" % (self.suffix, n_iter)

        self.work_subdir = os.path.join(self.work_dir, dname)
        self.v_tasks = [dname]

        # set restart wfn for DFT initial guess
        if n_iter > 0:
            wfn_dname = "search_%s.%06d" % (self.suffix, (n_iter - 1))
        else:
            wfn_dname = "ref_%s" % self.suffix
        wfn_restart = os.path.join(self.work_dir, wfn_dname,
                                   "cp2k-RESTART.wfn")

        # TODO: change the eps_scf
        kwargs.update({"eps_scf": 1e-2, "wfn_restart": wfn_restart})
        update_dict(fp_params,
                    self._water_pdos_input(n_wat=self.info_dict["n_wat"]))
        super().preset(
            pos_dielec=[L_VAC / 2.,
                        self.atoms.get_cell()[2][2] - L_VAC / 2.],
            fp_params=fp_params,
            calculate=calculate,
            **kwargs)

    def search_calculate(self):
        if not os.path.exists(os.path.join(self.work_subdir, "data.npy")):
            n_wat = self.info_dict["n_wat"]
            z_wat = self.atoms.get_positions()[self.info_dict["O_mask"],
                                               2] - self.info_dict["z_ave"]
            sort_ids = np.argsort(z_wat)
            cbm, vbm = self._water_mo_output(n_wat)

            np.save(os.path.join(self.work_subdir, "data.npy"),
                    [z_wat[sort_ids], cbm[sort_ids], vbm[sort_ids]])

        self.v_seq = [self._guess()]

    def preset(self, fp_params={}, calculate=False, **kwargs):
        v_start = kwargs.pop("v_start", -0.05 * (L_WAT + EPS_WAT * L_VAC * 2))
        v_end = kwargs.pop("v_end", 0.05 * (L_WAT + EPS_WAT * L_VAC * 2))
        n_step = kwargs.pop("n_step", 3)
        self.v_seq = np.linspace(v_start, v_end, n_step)
        self.v_seq += self.v_guess

        self.v_tasks = []
        for ii, v in enumerate(self.v_seq):
            efield = v / self.atoms.cell[2][2]
            self.v_tasks.append("task_%s.%06d" % (self.suffix, ii))
            logging.info("Macroscopic E-field: %.3f [V/A] in %s" %
                         (efield, self.v_tasks[-1]))

        dnames = glob.glob(
            os.path.join(self.work_dir, "search_%s.*" % self.suffix))
        dnames.sort()
        kwargs.update({
            "wfn_restart":
            os.path.join(self.work_dir, dnames[-1], "cp2k-RESTART.wfn")
        })

        update_dict(fp_params,
                    self._water_pdos_input(n_wat=self.info_dict["n_wat"]))

        super().preset(
            pos_dielec=[L_VAC / 2.,
                        self.atoms.get_cell()[2][2] - L_VAC / 2.],
            fp_params=fp_params,
            calculate=calculate,
            **kwargs)

    def calculate(self, **kwargs):
        super().calculate(pos_vac=0.75 * L_VAC,
                          save_fname="eps_data_%s" % self.suffix,
                          **kwargs)
        for dname in self.v_tasks:
            fname = os.path.join(self.work_dir, dname, "data.npy")
            if not os.path.exists(fname):
                n_wat = self.info_dict["n_wat"]
                z_wat = self.atoms.get_positions()[self.info_dict["O_mask"],
                                                   2] - self.info_dict["z_ave"]
                sort_ids = np.argsort(z_wat)
                cbm, vbm = self._water_mo_output(n_wat)

                np.save(os.path.join(self.work_subdir, "data.npy"),
                        [z_wat[sort_ids], cbm[sort_ids], vbm[sort_ids]])

    def workflow(self,
                 configs: str = "param.json",
                 ignore_finished_tag: bool = False):
        default_command = "mpiexec.hydra cp2k.popt"
        Eps.workflow(self, configs, default_command)

        # pbc: preset
        logging.info(
            "{:=^50}".format(" Start: set up files for PBC calculation "))
        tmp_params = self.wf_configs.get("pbc_preset", {})
        self.pbc_preset(calculate=True, **tmp_params)
        logging.info(
            "{:=^50}".format(" End: set up files for PBC calculation "))
        # pbc: DFT calculation
        self._dft_calculate(self.work_subdir, ignore_finished_tag)
        # pbc: calculate ref water MO
        logging.info("{:=^50}".format(" Start: analyse PBC data "))
        self.pbc_calculate()
        logging.info("{:=^50}".format(" End: analyse PBC data "))

        # ref: preset
        logging.info(
            "{:=^50}".format(" Start: set up files for dipole correction "))
        tmp_params = self.wf_configs.get("ref_preset", {})
        self.ref_preset(calculate=True, **tmp_params)
        logging.info(
            "{:=^50}".format(" End: set up files for dipole correction "))

        data_dict = {}
        for suffix in ["lo", "hi"]:
            self.suffix = suffix
            self.v_guess = 0.
            self.search_history = np.array([])
            data_dict[suffix] = {}

            dname = "ref_%s" % suffix
            self.work_subdir = os.path.join(self.work_dir, dname)
            # ref: DFT calculation
            self._dft_calculate(self.work_subdir, ignore_finished_tag)
            # ref: calculate dipole moment
            logging.info("{:=^50}".format(" Start: analyse %s data " % dname))
            tmp_params = self.wf_configs.get("ref_calculate", {})
            self.ref_calculate(**tmp_params)
            data_dict[suffix]["v_zero"] = self.v_zero
            logging.info("{:=^50}".format(" End: analyse %s data " % dname))

            convergence = self.wf_configs.get("convergence",
                                              SEARCH_CONVERGENCE)
            max_loop = self.wf_configs.get("max_loop", MAX_LOOP)
            search_flag = False
            for n_loop in range(max_loop):
                # search
                logging.info("{:=^50}".format(" Start: search %s iter.%06d " %
                                              (suffix, n_loop)))
                tmp_params = self.wf_configs.get("search_preset", {})
                self.search_preset(n_iter=n_loop, calculate=True, **tmp_params)
                # search: DFT calculation
                self._dft_calculate(self.work_subdir, ignore_finished_tag)
                self.search_calculate()
                # logging.info("Convergence [V]: %f" % self.convergence)
                logging.info("{:=^50}".format(" End: search %s iter.%06d " %
                                              (suffix, n_loop)))
                np.save(
                    os.path.join(self.work_dir,
                                 "search_history_%s.npy" % self.suffix),
                    self.search_history)
                if np.abs(self.convergence) <= convergence:
                    search_flag = True
                    logging.info("Finish searching in %d step(s)." %
                                 (n_loop + 1))
                    break
            if search_flag:
                self.v_guess = self.search_history[-1, 0]
            else:
                self.v_guess = self.search_history[
                    np.argmin(np.abs(self.search_history[:, 1])), 0]
                logging.warn("Cannot find converged Delta_V.")

            logging.info("{:=^50}".format(" Start: eps calculation "))
            # eps_cal: preset
            tmp_params = self.wf_configs.get("preset", {})
            self.preset(calculate=True, **tmp_params)
            data_dict[suffix]["v_cor"] = self.search_history[-1][0]
            # data_dict[suffix]["v_seq"] = self.v_seq
            # data_dict[suffix]["efield"] = self.v_seq / self.atoms.cell[2][2]
            # eps_cal: DFT calculation
            for task in self.v_tasks:
                self.work_subdir = os.path.join(self.work_dir, task)
                self._dft_calculate(self.work_subdir, ignore_finished_tag)
            self._load_data(fname="eps_data_%s" % self.suffix)
            tmp_params = self.wf_configs.get("calculate", {})
            self.calculate(**tmp_params)
            logging.info("{:=^50}".format(" End: eps calculation "))

        save_dict(data_dict, os.path.join(self.work_dir, "task_info.json"))
        self.make_plots()

    def _convert(self, inverse: bool = False):
        cell = self.pbc_atoms.get_cell()
        new_cell = self.pbc_atoms.get_cell()
        # add vac layer in both boundary of the cell
        new_cell[2][2] += 2 * L_VAC
        coords = self.pbc_atoms.get_positions()
        if inverse:
            coords[:, 2] = cell[2][2] - coords[:, 2]

        # fold Pt slab
        mask_z = (coords[:, 2] > cell[2][2] / 2.)
        mask = self.pbc_info["metal_mask"] * mask_z
        coords[mask, 2] -= cell[2][2]
        # shift supercell
        z_shifted = L_VAC - coords[:, 2].min()
        coords[:, 2] += z_shifted

        if inverse:
            z_ave = cell[2][2] - self.pbc_info["z_hi"] + z_shifted
        else:
            z_ave = self.pbc_info["z_lo"] + z_shifted

        # logging.info("Position of metal surface: %.3f [A]" % z_ave)
        mask_atype = self.pbc_info["O_mask"]
        mask_z = (coords[:, 2] <= (z_ave + L_WAT))
        mask = mask_atype * mask_z
        ids_O = np.arange(len(self.pbc_atoms))[mask]
        ids_Pt = np.arange(len(self.pbc_atoms))[self.pbc_info["metal_mask"]]
        ids_sel = np.concatenate([ids_O, ids_O + 1, ids_O + 2, ids_Pt])
        ids_sel = np.sort(ids_sel)

        new_atoms = Atoms(symbols=self.pbc_info["atype"][ids_sel],
                          positions=coords[ids_sel],
                          cell=new_cell,
                          pbc=True)

        # check water config
        check_water(new_atoms)

        return new_atoms

    def _water_mo_output(self, n_wat):
        cbm = []
        vbm = []
        for ii in range(n_wat):
            fname = os.path.join(self.work_subdir,
                                 "cp2k-list%d-1.pdos" % (ii + 1))
            task = Cp2kPdos(fname)
            cbm.append(task.cbm)
            vbm.append(task.vbm)
        return np.array(cbm), np.array(vbm)

    def _setup(self, type: str):
        setattr(IterElecEps, "%s_atoms" % type, self.atoms.copy())
        setattr(IterElecEps, "%s_info" % type, {})
        info_dict = getattr(self, "%s_info" % type)
        atoms = getattr(self, "%s_atoms" % type)
        info_dict["atype"] = np.array(atoms.get_chemical_symbols())

        info_dict["O_mask"] = (info_dict["atype"] == "O")
        info_dict["H_mask"] = (info_dict["atype"] == "H")
        info_dict["water_mask"] = info_dict["O_mask"] + info_dict["H_mask"]
        info_dict["metal_mask"] = ~info_dict["water_mask"]
        info_dict["n_wat"] = len(
            info_dict["atype"][info_dict["water_mask"]]) // 3

        logging.info("Number of water molecules: %d" % info_dict["n_wat"])

        getattr(self, "_%s_setup" % type)()

    def _pbc_setup(self):
        z = self.pbc_atoms.get_positions()[:, 2]
        mask_z = (z < self.pbc_atoms.get_cell()[2][2] / 2.)
        mask = mask_z * self.pbc_info["metal_mask"]
        self.pbc_info["z_lo"] = np.sort(z[mask])[-N_SURF:].mean()
        mask_z = (z > self.pbc_atoms.get_cell()[2][2] / 2.)
        mask = mask_z * self.pbc_info["metal_mask"]
        self.pbc_info["z_hi"] = np.sort(z[mask])[:N_SURF].mean()
        logging.info("Position of lower surface: %.3f [A]" %
                     self.pbc_info["z_lo"])
        logging.info("Position of upper surface: %.3f [A]" %
                     self.pbc_info["z_hi"])

    def _ref_lo_setup(self):
        mask = self.ref_lo_info["metal_mask"]
        z = self.ref_lo_atoms.get_positions()[mask, 2]
        self.ref_lo_info["z_ave"] = np.sort(z)[-N_SURF:].mean()
        logging.info("Position of metal surface: %.3f [A]" %
                     self.ref_lo_info["z_ave"])

    def _ref_hi_setup(self):
        mask = self.ref_hi_info["metal_mask"]
        z = self.ref_hi_atoms.get_positions()[mask, 2]
        self.ref_hi_info["z_ave"] = np.sort(z)[-N_SURF:].mean()
        logging.info("Position of metal surface: %.3f [A]" %
                     self.ref_hi_info["z_ave"])

    def make_plots(self, out=None, sigma=0.0):
        if not os.path.exists(os.path.join(self.work_dir, "figures")):
            os.makedirs(os.path.join(self.work_dir, "figures"))
        if out is None:
            out = [["hartree", "rho_pol"], ["delta_efield", "inveps"]]

        fnames = glob.glob(os.path.join(self.work_dir, "*.%s" % self.data_fmt))
        for data_fname in fnames:
            if not "task_info" in data_fname:
                # read data
                data_dict = load_dict(data_fname)[sigma]

                fig, axs = self._make_plots(out,
                                            data_dict,
                                            scale=(np.min(data_dict["efield"]),
                                                   np.max(
                                                       data_dict["efield"])))
                figure_name = os.path.splitext(os.path.basename(data_fname))[0]
                fig.savefig(os.path.join(self.work_dir, "figures",
                                         "%s.png" % figure_name),
                            bbox_inches='tight')
