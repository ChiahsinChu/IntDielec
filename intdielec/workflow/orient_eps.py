# SPDX-License-Identifier: LGPL-3.0-or-later
import MDAnalysis as mda
from ase import Atoms
from MDAnalysis import transformations as trans
from toolbox import plot
from toolbox.plot.figure import Figure, FullCellFigure
from toolbox.utils.math import *
from toolbox.utils.unit import *
from toolbox.utils.utils import load_dict, safe_makedirs

from ..utils import *
from ..watanalysis.dielectric import InverseDielectricConstant
from . import Eps

plot.use_style("pub")


class OrientEps(Eps):
    def __init__(
        self,
        work_dir: str = ".",
        topo: str = None,
        coord: str = None,
        data_fmt: str = "pkl",
        dimensions=None,
        **kwargs,
    ) -> None:
        super().__init__(work_dir, data_fmt)

        if topo is None:
            self.topo = os.path.join(work_dir, "system.data")
        else:
            self.topo = topo
        if coord is None:
            self.coord = os.path.join(work_dir, "dump.lammpstrj")
        else:
            self.coord = coord

        kwargs.update(
            {
                "topology_format": os.path.splitext(self.topo)[-1][1:].upper(),
                "format": os.path.splitext(self.coord)[-1][1:].upper(),
            }
        )
        if kwargs["format"] == "LAMMPSTRJ":
            kwargs["format"] = "LAMMPSDUMP"
        if kwargs["topology_format"] == "DATA":
            kwargs.update({"atom_style": self._get_atom_style()})

        self.universe = mda.Universe(self.topo, self.coord, **kwargs)
        if dimensions is not None:
            transform = trans.boxdimensions.set_dimensions(dimensions)
            self.universe.trajectory.add_transformations(transform)
        self.water = self.universe.select_atoms("name O or name H")

    def workflow(
        self, start=0, stop=None, step=1, IDC=InverseDielectricConstant, **kwargs
    ):
        if stop is None:
            stop = self.universe._trajectory.n_frames

        task = IDC(atomgroups=self.water, **kwargs)
        task.run(start, stop, step)
        self.results = task.results
        self._save_data()
        self.make_plots()

    def make_plots(self):
        if not hasattr(self, "results"):
            self.results = load_dict(self.work_dir, "eps_data.%s" % self.data_fmt)

        work_dir = os.path.join(self.work_dir, "figures")
        safe_makedirs(work_dir)

        xlim = (0, self.universe._trajectory.ts.dimensions[2])
        # inveps
        figure = FullCellFigure()
        figure.setup(
            self.results["bins"],
            self.results["inveps"],
            xlim=xlim,
            z_surfs=(self.results["z_lo"], self.results["z_hi"]),
            color="blue",
        )
        figure.set_labels("orient_inveps")
        figure.fig.savefig(os.path.join(work_dir, "orient_inveps.png"))

        # local polarization distribution
        figure = FullCellFigure()
        figure.setup(
            self.results["bins"],
            self.results["m"],
            xlim=xlim,
            z_surfs=(self.results["z_lo"], self.results["z_hi"]),
            color="blue",
        )
        figure.set_labels("polarization")
        figure.fig.savefig(os.path.join(work_dir, "polarization.png"))

        # eps_ave
        lz = self.universe._trajectory.ts.dimensions[2] / 2 - self.results["z_lo"]
        xlim = (0, lz)
        x = np.linspace(0, lz)
        xp = self.results["bins"] - self.results["z_lo"]
        fp = self.results["inveps"]
        y_lo = np.interp(x, xp, fp)
        xp = np.sort(self.results["z_hi"] - self.results["bins"])
        fp = np.flip(self.results["inveps"])
        y_hi = np.interp(x, xp, fp)
        inveps = (y_lo + y_hi) / 2
        ave_eps = 1.0 / cumave(inveps)

        figure = Figure()
        ax = figure.ax

        color = "blue"
        figure.setup(x, inveps, xlim=xlim, color=color)
        figure.set_labels("orient_inveps")
        ax.tick_params(axis="y", labelcolor=color)
        ax.axhline(y=0.0, color=color, ls="--", alpha=0.5)
        ax.axhline(y=1.0, color=color, ls="--", alpha=0.5)

        ax_right = ax.twinx()
        color = "red"
        ax_right.plot(x, ave_eps, color=color)
        ax_right.set_ylim(bottom=0.0)
        ax_right.tick_params(axis="y", labelcolor=color)
        ax_right.axhline(y=1.0, color=color, ls="--", alpha=0.5)
        ax_right.axvline(x=6.0, color=color, ls="--", alpha=0.5)
        ax_right.set_ylabel(r"$\varepsilon_{ave}$", color=color)

        figure.fig.savefig(os.path.join(work_dir, "eps_ave.png"))

        # differential capacitance
        const = (
            constants.epsilon_0
            / constants.micro
            * constants.centi
            / (constants.angstrom / constants.centi)
        )

        figure = Figure()
        ax = figure.ax

        color = "blue"
        figure.setup(x, inveps, xlim=xlim, color=color)
        figure.set_labels("orient_inveps")
        ax.tick_params(axis="y", labelcolor=color)
        ax.axhline(y=0.0, color=color, ls="--", alpha=0.5)
        ax.axhline(y=1.0, color=color, ls="--", alpha=0.5)

        ax_right = ax.twinx()
        color = "red"
        c_d = handle_zero_division(ave_eps * const, x, threshold=0.1)
        ax_right.plot(x, c_d, color=color)
        ax_right.set_ylim(0.0, np.min([c_d.max(), 20]))
        ax_right.tick_params(axis="y", labelcolor=color)
        ax_right.axhline(y=20.0, color=color, ls="--", alpha=0.5)
        ax_right.axvline(x=6.0, color=color, ls="--", alpha=0.5)
        ax_right.set_ylabel(r"${\rm C_d}$ [$\mu F/cm^2$]", color=color)

        figure.fig.savefig(os.path.join(work_dir, "differential_capacitance.png"))

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
        atoms = Atoms(
            symbols=self.universe.atoms.types,
            positions=self.universe._trajectory.ts.positions,
            pbc=True,
        )
        atoms.set_cell(self.universe._trajectory.ts.dimensions)
        return atoms
