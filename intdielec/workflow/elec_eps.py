import copy
import glob
import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from ase import Atoms, io
from scipy import stats

from .. import plot
from ..calculator.cp2k import Cp2kCalculator
from ..io.cp2k import (Cp2kCube, Cp2kHartreeCube, Cp2kInput, Cp2kOutput,
                       Cp2kPdos)
from ..io.template import cp2k_default_input
from ..utils.config import check_water
from ..utils.math import *
from ..utils.unit import *
from ..utils.utils import load_dict, save_dict, update_dict
from . import Eps

_EPSILON = VAC_PERMITTIVITY / UNIT_CHARGE * ANG_TO_M
EPS_VAC = 1.
EPS_INT = 4.
EPS_WAT = 2.

N_SURF = 16
L_VAC = 15.
L_INT = 5.
L_QM_WAT = 15.
L_MM_WAT = 10.
L_WAT_PDOS = 10.
MAX_LOOP = 10
SEARCH_CONVERGENCE = 1e-2
SLOPE = 1. / 0.0765

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

        logging.info("Number of atoms: %d" % len(atoms))
        self._setup()

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
            DeltaV = -output.potdrop[0]
        except:
            assert (vac_region is not None)
            fname = glob.glob(os.path.join(dname, "*hartree*.cube"))
            assert len(fname) == 1
            cube = Cp2kHartreeCube(fname[0], vac_region)
            output = cube.get_ave_cube()
            DeltaV = cube.potdrop
        self.set_v_zero(DeltaV)
        logging.info("Potential drop for zero E-field: %f [V]" % DeltaV)

    def preset(self,
               pos_dielec,
               fp_params={},
               calculate=False,
               pdos=True,
               **kwargs):
        update_d = {
            "dip_cor": False,
            "hartree": True,
            "eden": True,
            "totden": True,
            "extended_fft_lengths": True
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
        if pdos:
            update_dict(fp_params,
                        self._water_pdos_input(n_wat=self.info["n_wat"]))

        for v, task in zip(self.v_seq, self.v_tasks):
            dname = os.path.join(self.work_dir, task)
            if not os.path.exists(dname):
                os.makedirs(dname)
            if not os.path.exists(os.path.join(dname, "cp2k-RESTART.wfn")):
                if not hasattr(self, "suffix"):
                    wfn_restart = os.path.join(self.work_dir, "ref",
                                               "cp2k-RESTART.wfn")
                kwargs.update({"wfn_restart": wfn_restart})
            else:
                kwargs.update(
                    {"wfn_restart": os.path.join(dname, "cp2k-RESTART.wfn")})

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
            - [x] mo

            - [x] v_prime
            - [x] rho_pol
            - [x] efield
            - [x] polarization
            - [x] inveps
            - [x] std_inveps
        """
        logging.info("Vaccum position for E-field reference: %.2f [A]" %
                     pos_vac)
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

                fname = os.path.join(dname, "data.npy")
                if not os.path.exists(fname):
                    n_wat = self.info["n_wat"]
                    z_wat = self.atoms.get_positions()[self.info["O_mask"],
                                                       2] - self.info["z_ave"]
                    sort_ids = np.argsort(z_wat)
                    cbm, vbm = self._water_mo_output(dname, n_wat)

                    np.save(os.path.join(dname, "data.npy"),
                            [z_wat[sort_ids], cbm[sort_ids], vbm[sort_ids]])

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

    def make_plots(self, out=None, sigma=0.0, figure_name_suffix=""):
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
                fig.savefig(os.path.join(
                    self.work_dir, "figures",
                    "%s%s.png" % (figure_name, figure_name_suffix)),
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
            "lin_test": r"$\sigma(\varepsilon_e^{-1})$",
            "pdos": r"$E-E_F$ [eV]"
        }
        v_primes = list(data_dict.keys())
        v_primes.sort()
        shape = np.shape(out)
        nrows = shape[0]
        ncols = shape[1]
        fig, axs = plt.subplots(nrows=nrows,
                                ncols=ncols,
                                figsize=[ncols * 5, nrows * 3],
                                dpi=300)
        axs = np.reshape(axs, [nrows, ncols])

        xlabel = r"$z$ [A]"
        for ii in range(nrows):
            for jj in range(ncols):
                ax = axs[ii][jj]
                kw = out[ii][jj]
                try:
                    ylabel = ylabels_dict[kw]
                except:
                    raise AttributeError("Unknown keyword %s" % kw)

                if kw == "pdos":
                    self._make_plots_pdos(ax, data_dict, scale)
                    plot.ax_setlabel(ax, xlabel, ylabel)
                    ax.axhline(y=0., color="gray")
                    ax.axvline(x=self.l_vac, color="gray")
                    ax.axvline(x=np.max(xs[0]) - self.l_vac, color="gray")
                    ax.axvline(x=np.max(xs[0]) - self.l_vac - self.l_qm_wat,
                               color="gray")
                    ax.set_xlim(np.min(xs[0]), np.max(xs[0]))
                else:
                    ys = data_dict[kw]
                    shape = np.shape(ys)
                    if len(data_dict["v"]) == shape[0]:
                        labels = data_dict["efield"]
                    else:
                        labels = (np.array(data_dict["efield"])[1:] +
                                  np.array(data_dict["efield"])[:-1]) / 2.
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
                    # dielectric region
                    ax.axvline(x=5., color="gray")
                    ax.axvline(x=np.max(xs[0]) - 5., color="gray")

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

    def _make_plots_pdos(ax, data_dict, labels, scale):
        ys = data_dict["pdos"]
        xs = np.tile(data_dict["pdos_grid"], (len(ys), 1))
        plot.ax_colormap_lines(ax,
                               xs,
                               ys,
                               labels=labels,
                               scale=scale,
                               colormap="coolwarm")
        ax.plot(data_dict["ref_pdos_grid"],
                data_dict["ref_pdos"],
                color="black",
                label="ref")
        try:
            ax.plot(data_dict["pbc_pdos_grid"],
                    data_dict["pbc_pdos"],
                    color="gray",
                    label="pbc")
        except:
            pass

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

        self.make_plots()
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

    @staticmethod
    def _water_mo_output(work_dir, n_wat):
        cbm = []
        vbm = []
        for ii in range(n_wat):
            fname = os.path.join(work_dir, "cp2k-list%d-1.pdos" % (ii + 1))
            task = Cp2kPdos(fname)
            cbm.append(task.cbm)
            vbm.append(task.vbm)
        return np.array(cbm), np.array(vbm)

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

        z = atoms.get_positions()[info_dict["metal_mask"], 2]
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
        logging.info("Number of atoms: %d" % len(atoms))

        self._setup("pbc")

        self.v_ref = 0.

    def pbc_preset(self, fp_params={}, dname="pbc", calculate=False, **kwargs):
        kwargs.update({"dip_cor": False, "hartree": True})

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

        cbm, vbm = self._water_mo_output(self.work_subdir, n_wat)

        np.save(os.path.join(self.work_subdir, "data.npy"), [
            z_wat_vs_lo[sort_ids], z_wat_vs_hi[sort_ids], cbm[sort_ids],
            vbm[sort_ids]
        ])

        cube = Cp2kHartreeCube(
            os.path.join(self.work_subdir, "cp2k-v_hartree-1_0.cube"))
        pbc_hartree = cube.get_ave_cube()
        cp2k_out = Cp2kOutput(os.path.join(self.work_subdir, "output.out"))
        self.pbc_hartree = [pbc_hartree[0], pbc_hartree[1] - cp2k_out.fermi]

    def ref_preset(self, fp_params={}, calculate=False, **kwargs):
        self.work_subdir = os.path.join(self.work_dir, "ref_%s" % self.suffix)
        self.atoms = self._convert(self.l_qm_wat)
        self._setup("ref_%s" % self.suffix)
        # self.info = getattr(self, "ref_%s_info" % self.suffix)
        super().ref_preset(fp_params=fp_params,
                           dname="ref_%s" % self.suffix,
                           calculate=calculate,
                           **kwargs)

    def ref_calculate(self, vac_region=None):
        dname = "ref_%s" % self.suffix
        self.info = getattr(self, "%s_info" % dname)
        self.work_subdir = os.path.join(self.work_dir, dname)
        self.atoms = getattr(self, "%s_atoms" % dname)
        super().ref_calculate(vac_region=vac_region, dname=dname)
        self.info["v_zero"] = self.v_zero
        if not os.path.exists(os.path.join(self.work_subdir, "data.npy")):
            n_wat = self.info["n_wat"]
            z_wat = self.atoms.get_positions()[self.info["O_mask"],
                                               2] - self.info["z_ave"]
            sort_ids = np.argsort(z_wat)
            cbm, vbm = self._water_mo_output(self.work_subdir, n_wat)

            np.save(os.path.join(self.work_subdir, "data.npy"),
                    [z_wat[sort_ids], cbm[sort_ids], vbm[sort_ids]])
        self.v_seq = [self._guess()]

    def search_preset(self, n_iter, fp_params={}, calculate=False, **kwargs):
        dname = "search_%s.%06d" % (self.suffix, n_iter)

        self.work_subdir = os.path.join(self.work_dir, dname)
        self.v_tasks = [dname]

        if not os.path.exists(
                os.path.join(self.work_subdir, "cp2k-RESTART.wfn")):
            # set restart wfn for DFT initial guess
            if n_iter > 0:
                wfn_dname = "search_%s.%06d" % (self.suffix, (n_iter - 1))
            else:
                wfn_dname = "ref_%s" % self.suffix
            wfn_restart = os.path.join(self.work_dir, wfn_dname,
                                       "cp2k-RESTART.wfn")
            kwargs.update({"wfn_restart": wfn_restart})
        else:
            kwargs.update({
                "wfn_restart":
                os.path.join(self.work_subdir, "cp2k-RESTART.wfn")
            })

        update_dict(fp_params,
                    self._water_pdos_input(n_wat=self.info["n_wat"]))
        super().preset(pos_dielec=[
            self.l_vac / 2.,
            self.atoms.get_cell()[2][2] - self.l_vac / 2.
        ],
                       fp_params=fp_params,
                       calculate=calculate,
                       **kwargs)

    def search_calculate(self):
        if not os.path.exists(os.path.join(self.work_subdir, "data.npy")):
            n_wat = self.info["n_wat"]
            z_wat = self.atoms.get_positions()[self.info["O_mask"],
                                               2] - self.info["z_ave"]
            sort_ids = np.argsort(z_wat)
            cbm, vbm = self._water_mo_output(self.work_subdir, n_wat)

            np.save(os.path.join(self.work_subdir, "data.npy"),
                    [z_wat[sort_ids], cbm[sort_ids], vbm[sort_ids]])

        self.v_seq = [self._guess()]

    def preset(self, fp_params={}, calculate=False, **kwargs):
        v_start = kwargs.pop(
            "v_start", -0.01 * (self.l_qm_wat + EPS_WAT * self.l_vac * 2))
        v_end = kwargs.pop("v_end",
                           0.01 * (self.l_qm_wat + EPS_WAT * self.l_vac * 2))
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

        super().preset(pos_dielec=[5., self.atoms.get_cell()[2][2] - 5.],
                       fp_params=fp_params,
                       calculate=calculate,
                       **kwargs)

    def calculate(self, **kwargs):
        super().calculate(pos_vac=5.0 + 0.5 * (self.l_vac - 5.0),
                          save_fname="eps_data_%s" % self.suffix,
                          **kwargs)

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

    def workflow(self,
                 configs: str = "param.json",
                 ignore_finished_tag: bool = False):
        default_command = "mpiexec.hydra cp2k.popt"
        Eps.workflow(self, configs, default_command)

        self.l_qm_wat = self.wf_configs.get("l_qm_wat", L_QM_WAT)
        self.l_wat_pdos = self.wf_configs.get("l_wat_pdos", L_WAT_PDOS)
        self.l_vac = self.wf_configs.get("l_vac", L_VAC)
        self.l_mm_wat = self.wf_configs.get("l_mm_wat", L_MM_WAT)
        # self.n_surf = self.wf_configs.get("n_surf", N_SURF)

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

        data_dict = {}
        for suffix in ["lo", "hi"]:
            self.suffix = suffix
            self.v_guess = 0.
            self.search_history = np.array([])
            data_dict[suffix] = {}

            # ref: preset
            logging.info("{:=^50}".format(
                " Start: set up files for dipole correction "))
            tmp_params = self.wf_configs.get("ref_preset", {})
            self.ref_preset(calculate=True, **tmp_params)
            logging.info(
                "{:=^50}".format(" End: set up files for dipole correction "))
            # ref: DFT calculation
            self._dft_calculate(self.work_subdir, ignore_finished_tag)
            # ref: calculate dipole moment
            logging.info("{:=^50}".format(" Start: analyse ref_%s data " %
                                          suffix))
            tmp_params = self.wf_configs.get("ref_calculate", {})
            self.ref_calculate(**tmp_params)
            data_dict[suffix]["v_zero"] = self.v_zero
            logging.info("{:=^50}".format(" End: analyse ref_%s data " %
                                          suffix))

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
            data_dict[suffix]["z_ave"] = self.info["z_ave"]
            data_dict[suffix]["v_seq"] = self.v_seq.tolist()
            data_dict[suffix]["efield"] = (np.array(self.v_seq) /
                                           self.atoms.cell[2][2]).tolist()
            # eps_cal: DFT calculation
            for task in self.v_tasks:
                self.work_subdir = os.path.join(self.work_dir, task)
                self._dft_calculate(self.work_subdir, ignore_finished_tag)
            self._load_data(fname="eps_data_%s" % self.suffix)
            tmp_params = self.wf_configs.get("calculate", {})
            self.calculate(**tmp_params)
            logging.info("{:=^50}".format(" End: eps calculation "))

        data_dict["pbc"] = {
            "z_lo": self.pbc_info["z_lo"],
            "z_hi": self.pbc_info["z_hi"]
        }
        save_dict(data_dict, os.path.join(self.work_dir, "task_info.json"))
        self.make_plots()

    def _convert(self, l_wat):
        if self.suffix == "hi":
            inverse = True
        else:
            inverse = False

        cell = self.pbc_atoms.get_cell()
        new_cell = self.pbc_atoms.get_cell()
        # add vac layer in both boundary of the cell
        new_cell[2][2] += 2 * self.l_vac
        coords = self.pbc_atoms.get_positions()
        if inverse:
            coords[:, 2] = cell[2][2] - coords[:, 2]

        # fold Pt slab
        mask_z = (coords[:, 2] > cell[2][2] / 2.)
        mask = self.pbc_info["metal_mask"] * mask_z
        coords[mask, 2] -= cell[2][2]
        # shift supercell
        z_shifted = self.l_vac - coords[:, 2].min()
        coords[:, 2] += z_shifted

        if inverse:
            z_ave = cell[2][2] - self.pbc_info["z_hi"] + z_shifted
        else:
            z_ave = self.pbc_info["z_lo"] + z_shifted

        # logging.info("Position of metal surface: %.3f [A]" % z_ave)
        mask_atype = self.pbc_info["O_mask"]
        mask_z = (coords[:, 2] <= (z_ave + l_wat))
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

    def _setup(self, type: str):
        setattr(IterElecEps, "%s_atoms" % type, self.atoms.copy())
        setattr(IterElecEps, "%s_info" % type, {})
        self.info = getattr(self, "%s_info" % type)
        self.info["atype"] = np.array(self.atoms.get_chemical_symbols())
        self.info["O_mask"] = (self.info["atype"] == "O")
        self.info["H_mask"] = (self.info["atype"] == "H")
        self.info["water_mask"] = self.info["O_mask"] + self.info["H_mask"]
        self.info["metal_mask"] = ~self.info["water_mask"]
        self.info["n_wat"] = np.count_nonzero(self.info["O_mask"])

        logging.info("Number of water molecules: %d" % self.info["n_wat"])

        getattr(self, "_%s_setup" % type)()

    def _pbc_setup(self):
        self.info = self.pbc_info

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
        self.info = getattr(self, "ref_%s_info" % self.suffix)
        # atoms = getattr(self, "ref_%s_atoms" % self.suffix)

        mask = self.info["metal_mask"]
        z = self.atoms.get_positions()[mask, 2]
        self.info["z_ave"] = np.sort(z)[-N_SURF:].mean()
        logging.info("Position of metal surface: %.3f [A]" %
                     self.info["z_ave"])

    def _ref_hi_setup(self):
        self._ref_lo_setup()

    def _guess(self):
        logging.info("V_guess [V]: %f" % self.v_guess)

        # ref_data = np.load(os.path.join(self.work_dir, "pbc/data.npy"))
        # test_data = np.load(os.path.join(self.work_subdir, "data.npy"))
        # ref_id = np.argmin(np.abs(test_data[0] - self.l_wat_pdos))
        # self.convergence = test_data[-1][:ref_id].mean()
        # if self.suffix == "lo":
        #     self.convergence -= ref_data[-1][:ref_id].mean()
        # else:
        #     self.convergence -= ref_data[-1][-ref_id:].mean()

        cube = Cp2kHartreeCube(
            os.path.join(self.work_subdir, "cp2k-v_hartree-1_0.cube"))
        _test_hartree = cube.get_ave_cube()
        cp2k_out = Cp2kOutput(os.path.join(self.work_subdir, "output.out"))

        grids = np.arange(0, self.l_wat_pdos, 0.1)

        fp = _test_hartree[1] - cp2k_out.fermi
        xp = _test_hartree[0] - self.info["z_ave"]
        test_hartree = np.interp(grids, xp, fp).mean()

        fp = self.pbc_hartree[1]
        xp = self.pbc_hartree[0] - self.pbc_info["z_%s" % self.suffix]
        if self.suffix == "hi":
            xp = -xp
            fp = fp[np.argsort(xp)]
            xp = np.sort(xp)
        ref_hartree = np.interp(grids, xp, fp).mean()
        self.convergence = test_hartree - ref_hartree
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

        guess_method = self.wf_configs.get("guess_method", "ols_cut")
        guess_setup = self.wf_configs.get("guess_setup", {})
        self.guess_slope = guess_setup.pop("slope", SLOPE)
        self.v_guess = getattr(self, "_guess_%s" % guess_method)(**guess_setup)
        return self.v_guess

    def _guess_simple(self):
        # z = self.atoms.get_positions()[:, 2]
        # mask_coord = (z >= (self.info["z_ave"] + self.l_wat_pdos))
        # mask = self.info["O_mask"] * mask_coord
        # sel_water_ids = np.arange(len(self.atoms))[mask] // 3
        # n_e = 0.
        # for ii in sel_water_ids:
        #     fname = os.path.join(self.work_subdir,
        #                          "cp2k-list%d-1.pdos" % (ii + 1))
        #     pdos = Cp2kPdos(fname)
        #     e = pdos.energies - pdos.fermi
        #     mask = ((e - 0.1) * (e + 0.1) < 0.)
        #     raw_dos = pdos._get_raw_dos("total")
        #     occupation = pdos.occupation
        #     n_e += (raw_dos[mask] * (2.0 - occupation[mask])).sum()
        # logging.debug("number of electron at wat/vac interface: %f" % n_e)

        # cross_area = np.linalg.norm(
        #     np.cross(self.atoms.cell[0], self.atoms.cell[1]))
        # logging.debug("cross area: %f" % cross_area)
        # v_guess = 2 * self.l_vac * (n_e / cross_area / _EPSILON)
        # logging.debug("V_guess (1): %f" % v_guess)

        # dielectrics
        # # slope = (EPS_WAT * self.l_vac * 2 + self.l_qm_wat) / (EPS_WAT * self.l_vac + self.l_wat_pdos)
        # # emprical values
        # slope = 0.5
        # ref_data = np.load(os.path.join(self.work_dir, "pbc/data.npy"))
        # test_data = np.load(os.path.join(self.work_subdir, "data.npy"))
        # ref_id = np.argmin(np.abs(test_data[0] - self.l_wat_pdos))
        # logging.debug("ref_id: %d" % ref_id)
        # test_homo = test_data[-1][(ref_id - 4):(ref_id + 1)].mean()
        # if self.suffix == "lo":
        #     ref_homo = ref_data[-1][(ref_id - 4):(ref_id + 1)].mean()
        # else:
        #     ref_homo = ref_data[-1][-(ref_id - 1):-(ref_id - 6)].mean()
        # delta_v = test_homo - ref_homo
        # v_guess += delta_v * slope
        v_guess = self.convergence * self.guess_slope
        # efield = get_efields(1.0, l=[self.l_vac, L_INT, self.l_qm_wat-L_INT, self.l_vac], eps=[EPS_VAC,EPS_INT, EPS_WAT,  EPS_VAC]):
        # logging.debug("V_guess (2): %f" % v_guess)
        return v_guess

    def _guess_ols(self):
        if len(self.search_history) < 2:
            return self._guess_simple()
        else:
            result = stats.linregress(x=self.search_history[:, 0],
                                      y=self.search_history[:, 1])
            v_guess = -result.intercept / result.slope
        return v_guess

    def _guess_ols_cut(self, nstep=4):
        if len(self.search_history) < 2:
            return self._guess_simple()
        else:
            l_cut = min(len(self.search_history), nstep)
            result = stats.linregress(x=self.search_history[-l_cut:, 0],
                                      y=self.search_history[-l_cut:, 1])
            v_guess = -result.intercept / result.slope
        return v_guess

    def _guess_wls(self):
        if len(self.search_history) < 2:
            return self._guess_simple()
        else:
            #fit linear regression model
            X = self.search_history[:, 0]
            y = self.search_history[:, 1]
            X = sm.add_constant(x)
            wt = np.exp(-np.array(y)**2 / 0.1)
            fit_wls = sm.WLS(y, X, weights=wt).fit()
            return -fit_wls.params[0] / fit_wls.params[1]

    def _guess_wls_cut(self, nstep=4):
        if len(self.search_history) < 2:
            return self._guess_simple()
        else:
            l_cut = min(len(self.search_history), nstep)
            X = self.search_history[-l_cut:, 0]
            y = self.search_history[-l_cut:, 1]
            X = sm.add_constant(x)
            wt = np.exp(-np.array(y)**2 / 0.1)
            fit_wls = sm.WLS(y, X, weights=wt).fit()
            return -fit_wls.params[0] / fit_wls.params[1]


class QMMMIterElecEps(IterElecEps):
    """
    define the thickness of QM and MM water layer
    """
    def __init__(self,
                 atoms: Atoms = None,
                 work_dir: str = None,
                 data_fmt: str = "pkl") -> None:
        super().__init__(atoms, work_dir, data_fmt)

    def preset(self, fp_params={}, calculate=False, **kwargs):
        self.atoms = self._convert(self.l_qm_wat + self.l_mm_wat)

        v_start = kwargs.pop(
            "v_start", -0.01 * (self.l_qm_wat + EPS_WAT * self.l_vac * 2))
        v_end = kwargs.pop("v_end",
                           0.01 * (self.l_qm_wat + EPS_WAT * self.l_vac * 2))
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
        update_dict(fp_params, cp2k_default_input["qmmm"])
        update_dict(fp_params,
                    self._water_pdos_input_qmmm(n_wat=self.info["n_wat"]))
        update_dict(fp_params, self._qmmm_input())

        ElecEps.preset(self,
                       pos_dielec=[5., self.atoms.get_cell()[2][2] - 5.],
                       fp_params=fp_params,
                       calculate=calculate,
                       pdos=False,
                       **kwargs)

    def calculate(self, **kwargs):
        self.atoms = getattr(self, "ref_%s_atoms" % self.suffix)

        ElecEps.calculate(self,
                          pos_vac=5.0 + 0.5 * (self.l_vac - 5.0),
                          save_fname="eps_data_%s" % self.suffix,
                          **kwargs)
        # for dname in self.v_tasks:
        #     dname = os.path.join(self.work_dir, dname)
        #     fname = os.path.join(dname, "data.npy")
        #     if not os.path.exists(fname):
        #         n_wat = self.info["n_wat"]
        #         atoms = getattr(self, "ref_%s_atoms" % self.suffix)
        #         z_wat = atoms.get_positions()[self.info["O_mask"],
        #                                       2] - self.info["z_ave"]
        #         sort_ids = np.argsort(z_wat)
        #         cbm, vbm = self._water_mo_output(dname, n_wat)

        #         np.save(fname, [z_wat[sort_ids], cbm[sort_ids], vbm[sort_ids]])

    @staticmethod
    def _water_pdos_input_qmmm(n_wat):
        update_d = {"FORCE_EVAL": {"DFT": {"PRINT": {"PDOS": {"LDOS": []}}}}}
        for ii in range(n_wat):
            update_d["FORCE_EVAL"]["DFT"]["PRINT"]["PDOS"]["LDOS"].append({
                "LIST":
                "%d %d %d" % (ii + 1, n_wat + 2 * ii + 1, n_wat + 2 * ii + 2)
            })
        return update_d

    def _qmmm_input(self):
        atype = np.array(self.atoms.get_chemical_symbols())
        z = self.atoms.get_positions()[:, 2]
        O_ids = np.nonzero((atype == "O")
                           & (z < (self.info["z_ave"] + self.l_qm_wat)))[0] + 1
        # print(O_ids)
        H_ids = np.sort(np.concatenate([O_ids + 1, O_ids + 2]))
        # print(H_ids)
        Pt_ids = np.nonzero(atype == "Pt")[0] + 1

        update_d = {
            "FORCE_EVAL": {
                "SUBSYS": {
                    "TOPOLOGY": {
                        "GENERATE": {
                            "ISOLATED_ATOMS": {
                                "LIST": "1..%d" % len(self.atoms)
                            }
                        }
                    }
                },
                "QMMM": {
                    "QM_KIND": [{
                        "_":
                        "O",
                        "MM_INDEX":
                        "%s" %
                        np.array2string(O_ids, max_line_width=1000)[1:-1]
                    }, {
                        "_":
                        "H",
                        "MM_INDEX":
                        "%s" %
                        np.array2string(H_ids, max_line_width=1000)[1:-1]
                    }, {
                        "_":
                        "Pt",
                        "MM_INDEX":
                        "%s" %
                        np.array2string(Pt_ids, max_line_width=1000)[1:-1]
                    }]
                }
            }
        }
        return update_d
