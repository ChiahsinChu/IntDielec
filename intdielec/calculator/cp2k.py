import logging
import os
import sys
from ase import io

from ase.calculators.cp2k import CP2K

from ..io.cp2k import Cp2kOutput
from ..utils.utils import load_dict, iterdict


class Cp2kCalculator:
    def __init__(self, work_dir) -> None:
        self.work_dir = work_dir

    def run(self, type="bash", **kwargs):
        root_dir = os.getcwd()
        os.chdir(self.work_dir)
        logging.info("{:=^50}".format(" Start: CP2K calculation "))
        getattr(self, "run_%s" % type)(**kwargs)
        logging.info("{:=^50}".format(" End: CP2K calculation "))
        os.chdir(root_dir)

    def run_bash(self, command, inp="input.inp", out="output.out"):
        logging.info("Path: %s" % os.getcwd())
        if inp is not None:
            command += " %s " % inp
        if out is not None:
            command += "> %s" % out
        else:
            out = "output.out"
        os.system(command=command)
        try:
            output = Cp2kOutput(out)
            with open("finished_tag", 'w') as f:
                pass
        except:
            sys.exit("CP2K calculation is not finished!")

    def run_ase(self):
        atoms = io.read("coord.xyz")
        inp_dict = load_dict("input.json")
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

    @staticmethod
    def _dict_to_cp2k_input(input_dict):
        input_str = iterdict(input_dict, out_list=["\n"], loop_idx=0)
        s = "\n".join(input_str)
        s = s.strip("\n")
        return s
