import csv
import logging
import os
import pickle

import h5py

from ..utils.utils import read_json


class Eps:
    def __init__(
        self,
        work_dir: str = None,
        data_fmt: str = "pkl",
    ) -> None:
        self.work_dir = work_dir

        self.data_fmt = data_fmt
        self._load_data()

        logging.info("{:=^50}".format(" Start: Eps Calculation "))

    def workflow(self, configs: str = "param.json", default_command=None):
        self.wf_configs = read_json(configs)
        # set env variables
        load_module = self.wf_configs.get("load_module", [])
        command = ""
        if len(load_module) > 0:
            command = "module load "
            for m in load_module:
                command += (m + " ")
            command += "&& "
        _command = self.wf_configs.get("command", default_command)
        command += _command
        # command += " && touch finished_tag"
        self.command = command

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

    def _load_data_pkl(self, fname):
        with open(fname, "rb") as f:
            self.results = pickle.load(f)

    def _load_data_hdf5(self, fname):
        self.results = {}

    def _save_data(self):
        fname = os.path.join(self.work_dir, "eps_data.%s" % self.data_fmt)
        getattr(self, "_save_data_%s" % self.data_fmt)(fname)

    def _save_data_csv(self, fname):
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results.keys())
            writer.writeheader()
            writer.writerow(self.results)

    def _save_data_pkl(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self.results, f)

    def _save_data_hdf5(self, fname):
        with h5py.File(fname, "a") as f:
            n = len(f)
            dts = f.create_group("%02d" % n)
            for k, v in self.results.items():
                dts.create_dataset(k, data=v)


# class CP2K(_CP2K):
#     """
#     https://www.cp2k.org/tools:ase
#     https://wiki.fysik.dtu.dk/ase/ase/calculators/cp2k.html#ase.calculators.cp2k.CP2K

#     Allow keywords covered by inp
#     """

#     default_parameters = dict(auto_write=None,
#                               basis_set=None,
#                               basis_set_file=None,
#                               charge=None,
#                               cutoff=None,
#                               force_eval_method=None,
#                               inp='',
#                               max_scf=None,
#                               potential_file=None,
#                               pseudo_potential=None,
#                               stress_tensor=None,
#                               uks=None,
#                               poisson_solver=None,
#                               xc=None,
#                               print_level=None)

#     def _generate_input(self):
#         """Generates a CP2K input file"""
#         from ase.calculators.cp2k import parse_input, InputSection

#         p = self.parameters
#         root = parse_input(p.inp)

#         try:
#             if p.print_level:
#                 root.add_keyword('GLOBAL', 'PRINT_LEVEL ' + p.print_level)
#         except:
#             pass

#         try:
#             pass
#         except:
#             pass

#         try:
#             pass
#         except:
#             pass

#         try:
#             pass
#         except:
#             pass

#         try:
#             pass
#         except:
#             pass

#         try:
#             pass
#         except:
#             pass

#         try:
#             pass
#         except:
#             pass

#         try:
#             pass
#         except:
#             pass

#         if p.force_eval_method:
#             root.add_keyword('FORCE_EVAL', 'METHOD ' + p.force_eval_method)
#         if p.stress_tensor:
#             root.add_keyword('FORCE_EVAL', 'STRESS_TENSOR ANALYTICAL')
#             root.add_keyword('FORCE_EVAL/PRINT/STRESS_TENSOR',
#                              '_SECTION_PARAMETERS_ ON')
#         if p.basis_set_file:
#             root.add_keyword('FORCE_EVAL/DFT',
#                              'BASIS_SET_FILE_NAME ' + p.basis_set_file)
#         if p.potential_file:
#             root.add_keyword('FORCE_EVAL/DFT',
#                              'POTENTIAL_FILE_NAME ' + p.potential_file)
#         if p.cutoff:
#             root.add_keyword('FORCE_EVAL/DFT/MGRID',
#                              'CUTOFF [eV] %.18e' % p.cutoff)
#         if p.max_scf:
#             root.add_keyword('FORCE_EVAL/DFT/SCF', 'MAX_SCF %d' % p.max_scf)
#             root.add_keyword('FORCE_EVAL/DFT/LS_SCF', 'MAX_SCF %d' % p.max_scf)

#         if p.xc:
#             legacy_libxc = ""
#             for functional in p.xc.split():
#                 functional = functional.replace("LDA", "PADE")  # resolve alias
#                 xc_sec = root.get_subsection('FORCE_EVAL/DFT/XC/XC_FUNCTIONAL')
#                 # libxc input section changed over time
#                 if functional.startswith("XC_") and self._shell.version < 3.0:
#                     legacy_libxc += " " + functional  # handled later
#                 elif functional.startswith("XC_"):
#                     s = InputSection(name='LIBXC')
#                     s.keywords.append('FUNCTIONAL ' + functional)
#                     xc_sec.subsections.append(s)
#                 else:
#                     s = InputSection(name=functional.upper())
#                     xc_sec.subsections.append(s)
#             if legacy_libxc:
#                 root.add_keyword('FORCE_EVAL/DFT/XC/XC_FUNCTIONAL/LIBXC',
#                                  'FUNCTIONAL ' + legacy_libxc)

#         if p.uks:
#             root.add_keyword('FORCE_EVAL/DFT', 'UNRESTRICTED_KOHN_SHAM ON')

#         if p.charge and p.charge != 0:
#             root.add_keyword('FORCE_EVAL/DFT', 'CHARGE %d' % p.charge)

#         # add Poisson solver if needed
#         if p.poisson_solver == 'auto' and not any(self.atoms.get_pbc()):
#             root.add_keyword('FORCE_EVAL/DFT/POISSON', 'PERIODIC NONE')
#             root.add_keyword('FORCE_EVAL/DFT/POISSON', 'PSOLVER  MT')

#         # write coords
#         syms = self.atoms.get_chemical_symbols()
#         atoms = self.atoms.get_positions()
#         for elm, pos in zip(syms, atoms):
#             line = '%s %.18e %.18e %.18e' % (elm, pos[0], pos[1], pos[2])
#             root.add_keyword('FORCE_EVAL/SUBSYS/COORD', line, unique=False)

#         # write cell
#         pbc = ''.join([a for a, b in zip('XYZ', self.atoms.get_pbc()) if b])
#         if len(pbc) == 0:
#             pbc = 'NONE'
#         root.add_keyword('FORCE_EVAL/SUBSYS/CELL', 'PERIODIC ' + pbc)
#         c = self.atoms.get_cell()
#         for i, a in enumerate('ABC'):
#             line = '%s %.18e %.18e %.18e' % (a, c[i, 0], c[i, 1], c[i, 2])
#             root.add_keyword('FORCE_EVAL/SUBSYS/CELL', line)

#         # determine pseudo-potential
#         potential = p.pseudo_potential
#         if p.pseudo_potential == 'auto':
#             if p.xc and p.xc.upper() in (
#                     'LDA',
#                     'PADE',
#                     'BP',
#                     'BLYP',
#                     'PBE',
#             ):
#                 potential = 'GTH-' + p.xc.upper()
#             else:
#                 msg = 'No matching pseudo potential found, using GTH-PBE'
#                 warn(msg, RuntimeWarning)
#                 potential = 'GTH-PBE'  # fall back

#         # write atomic kinds
#         subsys = root.get_subsection('FORCE_EVAL/SUBSYS').subsections
#         kinds = dict([(s.params, s) for s in subsys if s.name == "KIND"])
#         for elem in set(self.atoms.get_chemical_symbols()):
#             if elem not in kinds.keys():
#                 s = InputSection(name='KIND', params=elem)
#                 subsys.append(s)
#                 kinds[elem] = s
#             if p.basis_set:
#                 kinds[elem].keywords.append('BASIS_SET ' + p.basis_set)
#             if potential:
#                 kinds[elem].keywords.append('POTENTIAL ' + potential)

#         output_lines = ['!!! Generated by ASE !!!'] + root.write()
#         return '\n'.join(output_lines)