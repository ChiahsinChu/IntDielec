# SPDX-License-Identifier: LGPL-3.0-or-later
from intdielec.workflow.orient_eps import OrientEps

eps_calc = OrientEps(
    work_dir="eps_calc",
    topo="eps_calc/interface.psf",
    coord="eps_calc/dump.xtc",
    dimensions=[11.246, 11.246, 35.94, 90, 90, 90],
)

eps_calc.make_plots()
