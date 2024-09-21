# SPDX-License-Identifier: LGPL-3.0-or-later
from toolbox.utils import *

from intdielec.workflow.elec_eps import FiniteFieldElecEps as EpsCalc

atoms = io.read("coord.xyz")
task = EpsCalc(atoms, work_dir=".", efield_seq=[-0.2, 0.0, 0.2])
task.workflow()
