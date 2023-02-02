import copy
import glob
import os
import re

import numpy as np
from ase import io
from ase.io.cube import read_cube_data

from ..utils.unit import *
from ..utils.utils import iterdict, update_dict, axis_dict
from .template import cp2k_default_input
"""
def iterdict(input_dict, out_list=["\n"], loop_idx=0):
    if len(out_list) == 0:
        out_list.append("\n")
    start_idx = len(out_list) - loop_idx - 2
    n_repeat = -1
    for k,v in input_dict.items():
        k=str(k) # cast key into string
        #if value is dictionary
        if isinstance(v, dict):
            out_list.insert(-1-loop_idx, "&"+k)
            out_list.insert(-1-loop_idx, "&END "+k)
            iterdict(v, out_list, loop_idx+1)
        elif isinstance(v, list):
            n_repeat = len(v)
            #print(loop_idx)
            #print(input_dict)
            #print(out_list)
            #print(n_repeat)
            break
        else:
            v = str(v)
            if k == "_":
                out_list[start_idx] = out_list[start_idx] + " " + v
            else: 
                out_list.insert(-1-loop_idx, k+" "+v)
                #out_list.insert(-1-loop_idx, v)
    if n_repeat >= 0 :
        end_str = out_list[-1-loop_idx]
        del out_list[-1-loop_idx]
        start_str = out_list[-1-loop_idx]
        del out_list[-1-loop_idx]
        for i in range(n_repeat):
            tmp_dict = {}
            for k, v in input_dict.items():
                k=str(k)
                tmp_dict[k] = v[i]
            out_list.insert(-loop_idx, start_str)
            out_list.insert(-loop_idx, end_str)
            iterdict(tmp_dict, out_list, loop_idx)
    return out_list
"""


class Cp2kInput():
    """
    Class for CP2K input file generation (on the basis of templates)
    
    Attributes
    ----------
    atoms: ASE Atoms object
        TBC
    input_type: str
        TBC
    pp_dir: str
        directory for basis set, peusudopotential, etc.
    wfn_restart: str
        wfn file for restart, see ref: 
    qm_charge: float
        charge in QS
    multiplicity: int
        ref:
    uks: boolen
        ref:
    cutoff: int
        ref: 
    rel_cutoff: int
        ref: 

    Examples
    --------
    >>> from ase import io
    >>> from zjxpack.io.cp2k import Cp2kInput
    >>> atoms = io.read("POSCAR")
    >>> input = Cp2kInput(atoms, 
    >>>                   pp_dir="/data/basis", 
    >>>                   hartree=True, 
    >>>                   eden=True)
    >>> input.write()
    """

    def __init__(self, atoms, input_type="energy", **kwargs) -> None:
        self.atoms = atoms
        self.input_dict = copy.deepcopy(cp2k_default_input[input_type])
        # print(kwargs)
        self.set_params(kwargs)

    def set_params(self, kwargs):
        for kw, value in kwargs.items():
            update_d = getattr(self, "set_%s" % kw)(value)
            update_dict(self.input_dict, update_d)

    def write(self, output_dir=".", fp_params={}):
        """
        generate coord.xyz and input.inp for CP2K calculation at output_dir

        Parameters
        ----------
        output_dir: str
            directory to store coord.xyz and input.inp
        fp_params: dict
            dict for updated parameters
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cell = self.atoms.get_cell()
        cell_a = np.array2string(cell[0])
        cell_a = cell_a[1:-1]
        cell_b = np.array2string(cell[1])
        cell_b = cell_b[1:-1]
        cell_c = np.array2string(cell[2])
        cell_c = cell_c[1:-1]

        user_config = fp_params
        cell_config = {
            "FORCE_EVAL": {
                "SUBSYS": {
                    "CELL": {
                        "A": cell_a,
                        "B": cell_b,
                        "C": cell_c
                    }
                }
            }
        }
        update_dict(self.input_dict, user_config)
        update_dict(self.input_dict, cell_config)
        #output list
        input_str = iterdict(self.input_dict, out_list=["\n"], loop_idx=0)
        #del input_str[0]
        #del input_str[-1]
        #print(input_str)
        str = "\n".join(input_str)
        str = str.strip("\n")

        io.write(os.path.join(output_dir, "coord.xyz"), self.atoms)
        with open(os.path.join(output_dir, "input.inp"), "w",
                  encoding='utf-8') as f:
            f.write(str)

    def set_project(self, project_name: str):
        update_d = {"GLOBAL": {"PROJECT": project_name}}
        return update_d

    def set_pp_dir(self, pp_dir):
        update_d = {
            "FORCE_EVAL": {
                "DFT": {
                    "BASIS_SET_FILE_NAME": [
                        os.path.join(pp_dir, "BASIS_MOLOPT"),
                        os.path.join(pp_dir, "BASIS_ADMM"),
                        os.path.join(pp_dir, "BASIS_ADMM_MOLOPT"),
                        os.path.join(pp_dir, "BASIS_MOLOPT-HSE06")
                    ],
                    "POTENTIAL_FILE_NAME":
                    os.path.join(pp_dir, "GTH_POTENTIALS"),
                    "XC": {
                        "vdW_POTENTIAL": {
                            "PAIR_POTENTIAL": {
                                "PARAMETER_FILE_NAME":
                                os.path.join(pp_dir, "dftd3.dat")
                            }
                        }
                    }
                }
            }
        }
        return update_d

    def set_wfn_restrat(self, wfn_file):
        update_d = {"FORCE_EVAL": {"DFT": {"WFN_RESTART_FILE_NAME": wfn_file}}}
        return update_d

    def set_qm_charge(self, charge):
        update_d = {"FORCE_EVAL": {"DFT": {"CHARGE": charge}}}
        return update_d

    def set_multiplicity(self, multiplicity):
        update_d = {"FORCE_EVAL": {"DFT": {"MULTIPLICITY": multiplicity}}}
        return update_d

    def set_uks(self, flag):
        if flag:
            update_d = {"FORCE_EVAL": {"DFT": {"UKS": ".TRUE."}}}
            return update_d
        else:
            return {}

    def set_cutoff(self, cutoff):
        update_d = {"FORCE_EVAL": {"DFT": {"MGRID": {"CUTOFF": cutoff}}}}
        return update_d

    def set_rel_cutoff(self, rel_cutoff):
        update_d = {
            "FORCE_EVAL": {
                "DFT": {
                    "MGRID": {
                        "REL_CUTOFF": rel_cutoff
                    }
                }
            }
        }
        return update_d

    def set_kp(self, kp_mp):
        update_d = {
            "FORCE_EVAL": {
                "DFT": {
                    "KPOINTS": {
                        "SCHEME MONKHORST-PACK":
                        "%d %d %d" % (kp_mp[0], kp_mp[1], kp_mp[2]),
                        "SYMMETRY":
                        "ON",
                        "EPS_GEO":
                        1.0E-8,
                        "FULL_GRID":
                        "ON",
                        "VERBOSE":
                        "ON",
                        "PARALLEL_GROUP_SIZE":
                        0
                    }
                }
            }
        }
        return update_d

    def set_dip_cor(self, flag):
        if flag:
            update_d = {
                "FORCE_EVAL": {
                    "DFT": {
                        "SURFACE_DIPOLE_CORRECTION": ".TRUE."
                    }
                }
            }
            return update_d
        else:
            return {}

    def set_eden(self, flag):
        if flag:
            update_d = {
                "FORCE_EVAL": {
                    "DFT": {
                        "PRINT": {
                            "E_DENSITY_CUBE": {
                                "ADD_LAST": "NUMERIC",
                                "STRIDE": "8 8 1"
                            }
                        }
                    }
                }
            }
            return update_d
        else:
            return {}

    def set_mo(self, flag):
        if flag:
            update_d = {
                "FORCE_EVAL": {
                    "DFT": {
                        "PRINT": {
                            "MO_CUBES": {
                                "ADD_LAST": "NUMERIC"
                            }
                        }
                    }
                }
            }
            return update_d
        else:
            return {}

    def set_pdos(self, flag):
        if flag:
            update_d = {
                "FORCE_EVAL": {
                    "DFT": {
                        "PRINT": {
                            "PDOS": {
                                "COMPONENTS": ".TRUE.",
                                "ADD_LAST": "NUMERIC",
                                "NLUMO": -1,
                                "COMMON_ITERATION_LEVELS": 0
                            }
                        }
                    }
                }
            }
            return update_d
        else:
            return {}

    def set_hartree(self, flag):
        if flag:
            update_d = {
                "FORCE_EVAL": {
                    "DFT": {
                        "PRINT": {
                            "V_HARTREE_CUBE": {
                                "ADD_LAST": "NUMERIC",
                                "STRIDE": "8 8 1"
                            }
                        }
                    }
                }
            }
            return update_d
        else:
            return {}

    def set_efield(self, flag):
        if flag:
            update_d = {
                "FORCE_EVAL": {
                    "DFT": {
                        "PRINT": {
                            "EFIELD_CUBE": {
                                "ADD_LAST": "NUMERIC",
                                "STRIDE": "8 8 1"
                            }
                        }
                    }
                }
            }
            return update_d
        else:
            return {}

    def set_extended_fft_lengths(self, flag):
        if flag:
            update_d = {"GLOBAL": {"EXTENDED_FFT_LENGTHS": ".TRUE."}}
            return update_d
        else:
            return {}

    def set_kind(self, kind_dict: dict):
        """
        Parameter
        ---------
        kind_dict: dict
            dict to update kind section, for example:
            {
                "S": {
                    "ELEMENT": "O",
                    "BASIS_SET": "DZVP-MOLOPT-SR-GTH", 
                    "BASIS_SET": "GTH-PBE-q6"
                },
                "Li": {
                    "ELEMENT": "H",
                    "BASIS_SET": "DZVP-MOLOPT-SR-GTH", 
                    "BASIS_SET": "GTH-PBE-q1"
                }
            }
        """

        update_d = {"FORCE_EVAL": {"SUBSYS": {}}}
        update_dict(self.input_dict, update_d)

        old_kind_list = self.input_dict["FORCE_EVAL"]["SUBSYS"].get("KIND", [])
        if len(old_kind_list) > 0:
            for k, v in kind_dict.items():
                tmp_dict = copy.deepcopy(v)
                tmp_dict.update({"_": k})
                flag = False
                for ii, item in enumerate(old_kind_list):
                    if k == item["_"]:
                        # print(v)
                        old_kind_list[ii] = tmp_dict
                        flag = True
                        break
                if flag == False:
                    old_kind_list.append(tmp_dict)
        return {}

    def set_restart(self, flag):
        if flag:
            update_d = {
                "EXT_RESTART": {
                    "RESTART_FILE_NAME":
                    "%s-1.restart" % self.input_dict["GLOBAL"]["PROJECT"]
                }
            }
            return update_d
        else:
            return {}

    def set_md_step(self, md_step):
        update_d = {"MOTION": {"MD": {"STEPS": md_step}}}
        return update_d

    def set_md_temp(self, md_temp):
        update_d = {"MOTION": {"MD": {"TEMPERATURE": md_temp}}}
        return update_d

    def set_md_timestep(self, md_timestep):
        update_d = {"MOTION": {"MD": {"TIMESTEP": md_timestep}}}
        return update_d


class Cp2kOutput():

    def __init__(self, fname="output*") -> None:
        output_file = glob.glob(fname)
        output_file.sort()
        self.output_file = output_file

    @property
    def worktime(self):
        worktime = []
        for f in self.output_file:
            start, end = self.grep_time(f)
            worktime.append(self.time_gap(start, end))
        return worktime

    @staticmethod
    def grep_time(output_file):
        """
        grep the time info from cp2k output file

        Return:
            float list of time ["hour", "minute", "second"]
        """
        time_info = "".join(
            os.popen("grep 'PROGRAM STARTED AT' " + output_file).readlines())
        time_info = time_info.replace('\n', ' ')
        time_info = time_info.split(' ')
        start = []
        # ["hour", "minute", "second"]
        data = time_info[-2].split(":")
        for item in data:
            start.append(float(item))

        time_info = "".join(
            os.popen("grep 'PROGRAM ENDED AT' " + output_file).readlines())
        time_info = time_info.replace('\n', ' ')
        time_info = time_info.split(' ')
        end = []
        data = time_info[-2].split(":")
        for item in data:
            end.append(float(item))
        return start, end

    @staticmethod
    def time_gap(start, end):
        """
        Args:
            start: float list for inital time 
            end: float list for final time 
            in ['hour','minute','second']
        Return:
            time consuming for calculation 
            in second
        """
        t_i = np.array(start)
        t_f = np.array(end)
        t = t_f - t_i
        # second
        if t[-1] < 0:
            t[-1] = t[-1] + 60
            t[-2] = t[-2] - 1
        # minute
        if t[-2] < 0:
            t[-2] = t[-2] + 60
            t[-3] = t[-3] - 1
        # hour
        if t[-3] < 0:
            t[-3] = t[-3] + 24
        worktime = t[-1] + t[-2] * 60 + t[-3] * 60 * 60
        return worktime

    @staticmethod
    def grep_text(fname, start_pattern, end_pattern):
        with open(fname, "r") as f:
            lines = f.readlines()
        start_pattern = re.compile(start_pattern)
        end_pattern = re.compile(end_pattern)

        flag = False
        data_lines = []
        nframe = 0
        for line in lines:
            line = line.strip('\n')
            if start_pattern.match(line):
                flag = True
            if end_pattern.match(line):
                assert flag is True, (flag,
                                      'No charge data is found in this file.')
                flag = False
                nframe += 1
            if flag is True:
                data_lines.append(line)
        return nframe, data_lines

    @property
    def coord(self):
        """
        get atomic coordinate from cp2k output

        Return:
            coord numpy array (n_atom, 3)
        """
        start_pattern = r' MODULE QUICKSTEP:  ATOMIC COORDINATES IN angstrom'
        end_pattern = r' SCF PARAMETERS'
        for f in self.output_file:
            nframe, data_lines = self.grep_text(f, start_pattern, end_pattern)
        data_lines = np.reshape(data_lines, (nframe, -1))

        data_list = []
        elem_list = []
        for line in data_lines[:, 4:-4].reshape(-1):
            line_list = line.split()
            data_list.append([
                float(line_list[4]),
                float(line_list[5]),
                float(line_list[6])
            ])
            elem_list.append(line_list[2])
        self.chemical_symbols = np.reshape(elem_list, (nframe, -1))[0]
        return np.reshape(data_list, (nframe, -1, 3))

    @property
    def force(self):
        """
        get atomic force from cp2k output

        Return:
            force numpy array (n_atom, 3)
        """
        start_pattern = r' ATOMIC FORCES in'
        end_pattern = r' SUM OF ATOMIC FORCES'
        for f in self.output_file:
            nframe, data_lines = self.grep_text(f, start_pattern, end_pattern)
        data_lines = np.reshape(data_lines, (nframe, -1))

        data_list = []
        for line in data_lines[:, 3:].reshape(-1):
            line_list = line.split()
            data_list.append([
                float(line_list[4]) * AU_TO_EV_EVERY_ANG,
                float(line_list[5]) * AU_TO_EV_EVERY_ANG,
                float(line_list[6]) * AU_TO_EV_EVERY_ANG
            ])
        return np.reshape(data_list, (nframe, -1, 3))

    @property
    def energy(self):
        data = "".join(
            os.popen("grep 'Total energy:' " +
                     self.output_file[0]).readlines())
        data = data.replace('\n', ' ')
        data = data.split(' ')
        return float(data[-2]) * AU_TO_EV

    @property
    def scf_loop(self):
        data = "".join(
            os.popen("grep '*** SCF run converged in' " +
                     self.output_file[0]).readlines())
        if len(data) == 0:
            return -1
        else:
            data = data.replace('\n', ' ')
            data = data.split(' ')
            return int(data[-4])

    @property
    def fermi(self):
        data = "".join(
            os.popen("grep 'Fermi' " + self.output_file[0]).readlines())
        data = data.replace('\n', ' ')
        data = data.split(' ')
        return float(data[-2]) * AU_TO_EV

    @property
    def m_charge(self):
        start_pattern = r'                     Mulliken Population Analysis'
        end_pattern = r' # Total charge '

        for f in self.output_file:
            nframe, data_lines = self.grep_text(f, start_pattern, end_pattern)
        data_lines = np.reshape(data_lines, (nframe, -1))

        data_list = []
        for line in data_lines[:, 3:].reshape(-1):
            line_list = line.split()
            data_list.append(float(line_list[-1]))
        return np.reshape(data_list, (nframe, -1))

    @property
    def h_charge(self):
        start_pattern = re.compile(
            r'                           Hirshfeld Charges')
        pass


class Cp2kCube():

    def __init__(self, fname) -> None:
        self.cube_data, self.atoms = read_cube_data(fname)

    def get_ave_cube(self, axis="z", gaussian_sigma=0.):
        if hasattr(self, 'axis') and self.axis == axis_dict[axis] and hasattr(
                self, 'ave_cube_data'):
            pass
        else:
            cell_param = self.atoms.cell.cellpar()
            self.axis = axis_dict[axis]
            # assert cell_param[-3:]
            self.ave_grid = np.arange(
                0, cell_param[self.axis],
                cell_param[self.axis] / self.cube_data.shape[self.axis])
            ave_axis = tuple(np.delete(np.arange(3), self.axis).tolist())
            self.ave_cube_data = np.mean(self.cube_data, axis=ave_axis)

        if gaussian_sigma > 0.:
            self.ave_cube_data_convolve = gaussian_convolve(
                self.ave_grid, self.ave_cube_data, gaussian_sigma)
        else:
            self.ave_cube_data_convolve = copy.deepcopy(self.ave_cube_data)

        return (self.ave_grid, self.ave_cube_data, self.ave_cube_data_convolve)


class Cp2kHartreeCube(Cp2kCube):

    def __init__(
        self,
        fname,
        vac_region: list[int] = None,
    ) -> None:
        super().__init__(fname)
        if vac_region:
            self.set_vac_region(vac_region)

    def set_vac_region(self, vac_region):
        assert len(vac_region) == 2
        self.vac_region = vac_region

    def get_ave_cube(self, axis="z", gaussian_sigma=0):
        # (self.ave_grid, self.ave_cube_data, self.ave_cube_data_convolve)
        output = super().get_ave_cube(axis, gaussian_sigma)
        self.ave_cube_data_convolve *= AU_TO_EV
        self.ave_cube_data *= AU_TO_EV
        return (self.ave_grid, self.ave_cube_data, self.ave_cube_data_convolve)

    @property
    def potdrop(self):
        start_id = np.argmin(np.abs(self.ave_grid - self.vac_region[0]))
        end_id = np.argmin(np.abs(self.ave_grid - self.vac_region[1]))
        if start_id > end_id:
            _data = np.append(self.ave_cube_data[start_id:],
                              self.ave_cube_data[:end_id])
        else:
            _data = np.array(
                self.ave_cube_data)[self.vac_region[0]:self.vac_region[1]]
        dev_data = np.diff(_data, axis=0)
        p_jump = np.argmax(np.abs(dev_data))
        return dev_data[p_jump]

    @property
    def dipole(self):
        d = -self.potdrop * self.cross_area * (VAC_PERMITTIVITY / UNIT_CHARGE /
                                               M_2_ANGSTROM)
        return d

    def set_cross_area(self, cross_area):
        self.cross_area = cross_area


def get_coords(pos_file="cp2k-pos-1.xyz"):
    traj = io.read(pos_file, index=":")
    coord = []
    energy = []
    for atoms in traj:
        coord.append(atoms.get_positions())
        if atoms.info.get('E'):
            energy.append(atoms.info['E'])
        else:
            energy.append(0.)
    energies = np.array(energy) * AU_TO_EV
    coords = np.reshape(coord, (len(traj), -1))
    return coords, energies


def get_forces(frc_file="cp2k-frc-1.xyz"):
    frcs = io.read(frc_file, index=":")
    force = []
    for atoms in frcs:
        force.append(atoms.get_positions())
    forces = np.reshape(force, (len(frcs), -1)) * AU_TO_EV_EVERY_ANG
    return forces


def get_energies(ener_file="cp2k-1.ener"):
    energies = np.loadtxt(ener_file)[:, 4]
    return energies * AU_TO_EV


def get_temps(ener_file="cp2k-1.ener"):
    temps = np.loadtxt(ener_file)[:, 3]
    return temps


def get_charge(output_file="output*", qtype="mulliken"):
    if qtype == "mulliken":
        start_pattern = re.compile(
            r'                     Mulliken Population Analysis')
    elif qtype == "hirshfeld":
        start_pattern = re.compile(
            r'                           Hirshfeld Charges')
    else:
        pass

    flag = False
    end_pattern = re.compile(r' # Total charge ')

    with open(output_file, "r") as f:
        lines = f.readlines()

    data_lines = []
    nframe = 0
    for line in lines:
        line = line.strip('\n')
        if start_pattern.match(line):
            flag = True
        if end_pattern.match(line):
            assert flag is True, (flag,
                                  'No charge data is found in this file.')
            flag = False
            nframe += 1
        if flag is True:
            data_lines.append(line)

    data_lines = np.reshape(data_lines, (nframe, -1))
    data_list = []
    for line in data_lines[:, 3:].reshape(-1):
        line_list = line.split()
        data_list.append(float(line_list[-1]))
    return np.reshape(data_list, (nframe, -1))


"""
smoothing func: Gaussian kernel
"""


def gaussian_kernel(bins, sigma):
    if sigma == 0:
        output = np.zeros_like(bins)
        one_id = np.where(a == 0.)[0][0]
        output[one_id] = 1
        return output
    elif sigma > 0:
        A = 1 / (sigma * np.sqrt(2 * np.pi))
        output = np.exp(-bins * bins / (2 * sigma**2))
        output *= A
        return output
    else:
        raise AttributeError("Sigma should be non-negative value.")


def gaussian_convolve(xs, ys, sigma):
    nbins = len(xs) - 1

    output = []
    for x in xs:
        bins = xs - x
        tmp_out = gaussian_kernel(bins, sigma)
        bin_width = bins[1:] - bins[:-1]
        output.append(
            np.sum(bin_width * ((tmp_out * ys)[1:] + (tmp_out * ys)[:-1]) / 2))
    return np.array(output)


def get_pdos(pdos_file, smearing_width=0.2):
    if isinstance(pdos_file, str):
        alpha = PDOS(pdos_file)
        npts = len(alpha.e)
        totalDOS = alpha.smearing(smearing_width)
        eigenvalues = np.linspace(min(alpha.e), max(alpha.e), npts)
    elif isinstance(pdos_file, list) and len(pdos_file) == 2:
        alpha = PDOS(pdos_file[0])
        beta = PDOS(pdos_file[1])
        npts = len(alpha.e)
        alpha_smeared = alpha.smearing(smearing_width)
        beta_smeared = beta.smearing(smearing_width)
        totalDOS = alpha_smeared + beta_smeared
        eigenvalues = np.linspace(min(alpha.e), max(alpha.e), npts)
    else:
        raise AttributeError('Unknown type of pdos file')
    output = np.transpose([eigenvalues, totalDOS])
    return output


class PDOS:
    """ Projected electronic density of states from CP2K output files

        Reference: https://wiki.wpi.edu/deskinsgroup/Density_of_States
        
        Attributes
        ----------
        atom: str 
            the name of the atom where the DoS is projected
        iterstep: int
            the iteration step from the CP2K job
        efermi: float
            the energy of the Fermi level [a.u]
        e: float
            (eigenvalue - efermi) in eV
        occupation: int
            1 for occupied state or 0 for unoccupied
        pdos: nested list of float
            projected density of states on each orbital for each eigenvalue
            [[s1, p1, d1,....], [s2, p2, d2,...],...]
            s: pdos in s orbitals
            p: pdos in p orbitals
            d: pdos in d orbitals
            .
            .
            .
        tpdos: list of float
            sum of all the orbitals PDOS
            
        Methods
        -------
        smearing(self,npts, width)
            return the smeared tpdos 
    """

    def __init__(self, infilename):
        """Read a CP2K .pdos file and build a pdos instance

        Parameters
        ----------
        infilename: str
            pdos output from CP2K. 

        """
        input_file = open(infilename, 'r')

        firstline = input_file.readline().strip().split()
        secondline = input_file.readline().strip().split()

        # Kind of atom
        self.atom = firstline[6]
        #iterationstep
        self.iterstep = int(firstline[12][:-1])  #[:-1] delete ","
        # Energy of the Fermi level
        self.efermi = float(firstline[15])

        # it keeps just the orbital names
        secondline[0:5] = []
        self.orbitals = secondline

        pdos_data = np.loadtxt(infilename)

        # lines = input_file.readlines()

        eigenvalue = pdos_data[:, 1]
        self.occupation = pdos_data[:, 2]
        self.tpdos = np.sum(pdos_data[:, 3:], axis=-1)
        self.e = (eigenvalue - self.efermi) * AU_TO_EV

    def __add__(self, other):
        """Return the sum of two PDOS objects"""
        return self.tpdos + other.tpdos

    def delta(self, emin, emax, npts, energy, width):
        """Return a delta-function centered at energy
        
        Parameters
        ----------
        emin: float
            minimun eigenvalue
        emax: float
            maximun eigenvalue
        npts: int
            Number of points in the smeared pdos
        energy: float
            energy where the gaussian is centered
        width: float
            dispersion parameter

        Return 
        ------
        delta: numpy array
            array of delta function values

        """

        energies = np.linspace(emin, emax, npts)
        x = -((energies - energy) / width)**2
        return np.exp(x) / (np.sqrt(np.pi) * width)

    def smearing(self, width):
        """Return a gaussian smeared DOS"""

        return gaussian_convolve(self.e, self.tpdos, width)
