import MDAnalysis as mda
from ase import Atoms
import logging

from ..exts.toolbox.toolbox.utils.math import *
from ..exts.toolbox.toolbox.utils.unit import *
from ..exts.toolbox.toolbox.utils.utils import safe_makedirs
from ..exts.toolbox.toolbox import plot
from ..utils import *
from ..watanalysis.dielectric import InverseDielectricConstant as IDC
from . import Eps

plot.use_style("pub")


class OrientEps(Eps):
    def __init__(self,
                 work_dir: str = ".",
                 topo: str = None,
                 coord: str = None,
                 data_fmt: str = "pkl",
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
        self.water = self.universe.select_atoms("name O or name H")

    def run(self, start=0, stop=None, step=1, **kwargs):
        if stop is None:
            stop = self.universe._trajectory.n_frames

        task = IDC(atomgroups=self.water, **kwargs)
        task.run(start, stop, step)
        self.results = task.results
        self._save_data()
        self.make_plots()

    def make_plots(self):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[5, 6], dpi=300)
        x = self.results["bins"]

        # local polarization distribution
        ax = axs[0]
        xlabel = " "
        ylabel = r"local polarization [eA$^{-2}$]"
        ax.axhline(y=0., color="gray")
        ax.plot(x, self.results["m"])
        ax.set_xlim(np.min(x), np.max(x))
        plot.ax_setlabel(ax, xlabel, ylabel)

        # inveps
        ax = axs[1]
        xlabel = r"z - z$_{surf}$ [A]"
        ylabel = r"$\varepsilon_{ori}^{-1}$"
        ax.axhline(y=0., color="gray")
        ax.axhline(y=1., color="gray", ls="--")
        ax.plot(x, self.results["inveps"])
        ax.set_xlim(np.min(x), np.max(x))
        plot.ax_setlabel(ax, xlabel, ylabel)

        fig.subplots_adjust(hspace=0.)
        fig.savefig(os.path.join(self.work_dir, "ori_eps_cal.png"),
                    bbox_inches='tight',
                    transparent=True)

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