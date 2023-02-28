import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from ase import io, Atoms
from ..watanalysis.dielectric import ParallelInverseDielectricConstant as PIDC

from ..plot import use_style
from ..utils.parallel import parallel_exec
from ..utils.math import *
from ..utils.unit import *
from . import Eps

use_style("pub")


class OrientEps(Eps):
    def __init__(self,
                 work_dir: str = ".",
                 topo: str = None,
                 coord: str = None,
                 data_fmt: str = "hdf5",
                 **kwargs) -> None:
        super().__init__(work_dir, data_fmt)

        if topo is None:
            self.topo = os.path.join(work_dir, "system.data")
        else:
            self.topo = topo
        if coord is None:
            self.coord = os.path.join(work_dir, "dump.lammpstrj")
        else:
            self.coord = coord

        kwargs.update({
            "topology_format":
            os.path.splitext(self.topo)[-1][1:].upper(),
            "format":
            os.path.splitext(self.coord)[-1][1:].upper()
        })
        if kwargs["format"] == "LAMMPSTRJ":
            kwargs["format"] = "LAMMPSDUMP"
        if kwargs["topology_format"] == "DATA":
            kwargs.update({"atom_style": self._get_atom_style()})

        self.universe = mda.Universe(self.topo, self.coord, **kwargs)
        self.n_frames = self.universe._trajectory.n_frames

    def run(self, start=0, stop=None, step=1, n_proc=20, **kwargs):
        """
        task = PIDC(
            universe=self.universe,
            bins=np.arange(0, 10, 0.2),
            axis="z",
            temperature=330,
            surf_ids=surf_ids,
            make_whole=False,
            c_ag="name O",
            select_all=True,
        )
        """
        if stop is None:
            stop = self.n_frames
        surf_ids = kwargs.get("surf_ids", None)
        if surf_ids is None:
            from zjxpack.postprocess.metal import ECMetal

            metal_type = kwargs.pop("metal_type")
            surf_atom_num = kwargs.pop("surf_atom_num")
            atoms = ECMetal(self.atoms,
                            metal_type=metal_type,
                            surf_atom_num=surf_atom_num)
            surf_ids = atoms.get_surf_idx()
            kwargs.update({"surf_ids": surf_ids})

        kwargs.update({
            "make_whole": False,
            "c_ag": "name O",
            "select_all": True
        })

        task = PIDC(universe=self.universe, **kwargs)
        parallel_exec(task.run,
                      start=start,
                      stop=stop,
                      step=step,
                      n_proc=n_proc)
        self.results = task.results
        self.results.update({"bins": kwargs["bins"]})
        # if not os.path.exists(os.path.join(dname, "dielectric")):
        #     os.makedirs(os.path.join(dname, "dielectric"))
        # np.save(os.path.join(dname, "dielectric/inveps.npy"),
        #         task.results["inveps"])
        # np.save(os.path.join(dname, "dielectric/M2.npy"), task.results["M2"])
        # np.save(os.path.join(dname, "dielectric/m.npy"), task.results["m"])
        # np.save(os.path.join(dname, "dielectric/mM.npy"), task.results["mM"])
        # np.save(os.path.join(dname, "dielectric/M.npy"), task.results["M"])
        self._save_data()

    def plot(self):
        pass

    def _get_atom_style(self):
        # atom_style="id resid type charge x y z"
        atom_style = ""
        atoms = io.read(self.topo, format="lammps-data")
        data = atoms.__dict__["arrays"]
        # hasattr(atoms, "arrays")
        if "id" in data:
            atom_style += "id "
        if "mol-id" in data:
            atom_style += "resid "
        if "type" in data:
            atom_style += "type "
        if "initial_charges" in data:
            atom_style += "charge "
        atom_style += "x y z"
        return atom_style

    @property
    def atoms(self):
        atoms = Atoms(symbols=self.universe.atoms.types,
                      positions=self.universe._trajectory.ts.positions,
                      pbc=True)
        atoms.set_cell(self.universe._trajectory.ts.dimensions)
        return atoms