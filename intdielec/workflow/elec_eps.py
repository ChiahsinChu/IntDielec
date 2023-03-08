import copy
import glob
import logging
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, io

from . import Eps
from ..io.cp2k import Cp2kCube, Cp2kHartreeCube, Cp2kInput, Cp2kOutput, Cp2kPdos
from ..plot import core, use_style
from ..utils.math import *
from ..utils.unit import *
from ..utils.utils import update_dict, iterdict, read_json

_EPSILON = VAC_PERMITTIVITY / UNIT_CHARGE * ANG_TO_M

use_style("pub")


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

        dname = os.path.join(self.work_dir, dname)
        if not os.path.exists(dname):
            os.makedirs(dname)

        task = Cp2kInput(self.atoms, **kwargs)
        task.write(output_dir=dname, fp_params=fp_params, save_dict=calculate)

    def ref_calculate(self, vac_region=None, dname="ref"):
        try:
            fname = glob.glob(os.path.join(dname, "output*"))
            assert len(fname) == 1
            output = Cp2kOutput(fname[0])
            DeltaV = output.potdrop[0]
        except:
            assert (vac_region is not None)
            dname = os.path.join(self.work_dir, dname)
            fname = glob.glob(os.path.join(dname, "*hartree*.cube"))
            assert len(fname) == 1
            cube = Cp2kHartreeCube(fname[0], vac_region)
            output = cube.get_ave_cube()
            DeltaV = cube.potdrop
        self.set_v_zero(DeltaV)

    def preset(self, pos_dielec, fp_params={}, calculate=False, **kwargs):
        update_d = {
            "dip_cor": False,
            "hartree": True,
            "eden": True,
            "extended_fft_lengths": True,
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
            try:
                fname = glob.glob(os.path.join(dname, "*TOTAL_DENSITY*.cube"))
                assert len(fname) == 1
                cube = Cp2kCube(fname[0])
            except:
                fname = glob.glob(
                    os.path.join(dname, "*ELECTRON_DENSITY*.cube"))
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
        self._save_data()

    def plot(self, sigma=0.0, fname="eps_cal.png"):
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
        core.ax_setlabel(axs[0][0], xlabel, ylabel)
        axs[0][0].axhline(y=0., ls="--", color="gray")

        ylabel = r"$\Delta P$ [a.u.]"
        axs[0][1].set_xlim(data_dict[v_prime]["polarization"][0].min(),
                           data_dict[v_prime]["polarization"][0].max())
        core.ax_setlabel(axs[0][1], xlabel, ylabel)
        axs[0][1].axhline(y=0., ls="--", color="gray")

        ylabel = r"$\Delta E_z$ [V/A]"
        axs[1][0].set_xlim(data_dict[v_prime]["efield"][0].min(),
                           data_dict[v_prime]["efield"][0].max())
        core.ax_setlabel(axs[1][0], xlabel, ylabel)
        axs[1][0].axhline(y=0., ls="--", color="gray")

        # ylabel = r"$\varepsilon_e^{-1}=\frac{\Delta E_z}{\Delta E_{z,vac}}$"
        ylabel = r"$\varepsilon_e^{-1}$"
        axs[1][1].set_xlim(data_dict[v_prime]["inveps"][0].min(),
                           data_dict[v_prime]["inveps"][0].max())
        core.ax_setlabel(axs[1][1], xlabel, ylabel)
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
        core.ax_setlabel(ax, xlabel, ylabel)

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
        default_command = "mpiexec.hydra cp2k_shell.popt"
        super().workflow(configs, default_command)

        # ref: preset
        tmp_params = self.wf_configs.get("ref_preset", {})
        self.ref_preset(calculate=True, **tmp_params)

        # ref: DFT calculation
        self._bash_cp2k_calculator(os.path.join(self.work_dir, "ref"),
                                   ignore_finished_tag)
        # self._ase_cp2k_calculator(os.path.join(self.work_dir, "ref"),
        #                           ignore_finished_tag)

        # ref: calculate dipole moment
        tmp_params = self.wf_configs.get("ref_calculate", {})
        self.ref_calculate(**tmp_params)

        # eps_cal: preset
        tmp_params = self.wf_configs.get("preset", {})
        self.preset(calculate=True, **tmp_params)
        # eps_cal: DFT calculation
        for task in self.v_tasks:
            self._bash_cp2k_calculator(os.path.join(self.work_dir, task),
                                       ignore_finished_tag)
            # self._ase_cp2k_calculator(os.path.join(self.work_dir, task),
            #                           ignore_finished_tag)

        # eps_cal: calculate eps
        tmp_params = self.wf_configs.get("calculate", {})
        self.calculate(**tmp_params)

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

    @staticmethod
    def _dict_to_cp2k_input(input_dict):
        input_str = iterdict(input_dict, out_list=["\n"], loop_idx=0)
        s = "\n".join(input_str)
        s = s.strip("\n")
        return s

    def _ase_cp2k_calculator(self, work_dir, ignore_finished_tag):

        from ase.calculators.cp2k import CP2K

        root_dir = os.getcwd()
        os.chdir(work_dir)

        if (ignore_finished_tag == True) or (os.path.exists("finished_tag")
                                             == False):
            atoms = io.read("coord.xyz")
            inp_dict = read_json("input.json")
            label = inp_dict["GLOBAL"].pop("PROJECT", "cp2k")
            inp_dict["FORCE_EVAL"]["SUBSYS"].pop("TOPOLOGY", None)
            inp = self._dict_to_cp2k_input(inp_dict)

            calc = CP2K(command=self.command,
                        inp=inp,
                        label=label,
                        force_eval_method=None,
                        print_level=None,
                        stress_tensor=None,
                        basis_set=None,
                        pseudo_potential=None,
                        basis_set_file=None,
                        potential_file=None,
                        cutoff=None,
                        max_scf=None,
                        xc=None,
                        uks=None,
                        charge=None,
                        poisson_solver=None)
            atoms.calc = calc
            atoms.get_potential_energy()
            logging.info("{:=^50}".format(" End: CP2K calculation "))

        os.chdir(root_dir)

    def _bash_cp2k_calculator(self, work_dir, ignore_finished_tag):
        root_dir = os.getcwd()
        os.chdir(work_dir)

        if (ignore_finished_tag == True) or (os.path.exists("finished_tag")
                                             == False):
            logging.info("{:=^50}".format(" Start: CP2K calculation "))
            os.system(self.command)
            try:
                output = Cp2kOutput("output.out")
                with open("finished_tag", 'w') as f:
                    pass
            except:
                sys.exit("CP2K task is not finished!")
            logging.info("{:=^50}".format(" End: CP2K calculation "))

        os.chdir(root_dir)
