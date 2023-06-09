import MDAnalysis as mda
from ase import Atoms
from MDAnalysis import transformations as trans

from ..exts.toolbox.toolbox import plot
from ..exts.toolbox.toolbox.plot.figure import FullCellFigure, HalfCellFigure
from ..exts.toolbox.toolbox.utils.math import *
from ..exts.toolbox.toolbox.utils.unit import *
from ..exts.toolbox.toolbox.utils.utils import safe_makedirs, load_dict
from ..utils import *
from ..watanalysis.dielectric import InverseDielectricConstant
from . import Eps

plot.use_style("pub")


class OrientEps(Eps):
    def __init__(self,
                 work_dir: str = ".",
                 topo: str = None,
                 coord: str = None,
                 data_fmt: str = "pkl",
                 dimensions=None, 
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
        if dimensions is not None:
            transform = trans.boxdimensions.set_dimensions(dimensions)
            self.universe.trajectory.add_transformations(transform)
        self.water = self.universe.select_atoms("name O or name H")

    def run(self,
            start=0,
            stop=None,
            step=1,
            IDC=InverseDielectricConstant,
            **kwargs):
        if stop is None:
            stop = self.universe._trajectory.n_frames

        task = IDC(atomgroups=self.water, **kwargs)
        task.run(start, stop, step)
        self.results = task.results
        self._save_data()
        self.make_plots()

    def make_plots(self, half=False):
        if not hasattr(self, "results"):
            self.results = load_dict(self.work_dir, "eps_data.%s" % self.data_fmt)

        work_dir = os.path.join(self.work_dir, "figures")
        safe_makedirs(work_dir)

        # inveps
        figure = FullCellFigure()
        figure.setup(self.results["bins"], 
                     self.results["inveps"], 
                     z_surfs=(self.results["z_lo"], self.results["z_hi"]), 
                     color="blue")
        figure.set_labels("orient_inveps")
        figure.fig.savefig(os.path.join(work_dir, "orient_inveps.png"))

        # local polarization distribution
        figure = FullCellFigure()
        figure.setup(self.results["bins"], 
                     self.results["m"], 
                     z_surfs=(self.results["z_lo"], self.results["z_hi"]), 
                     color="blue")
        figure.set_labels("polarization")
        figure.fig.savefig(os.path.join(work_dir, "polarization.png"))

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

