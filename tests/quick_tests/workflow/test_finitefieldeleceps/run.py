from intdielec.workflow.elec_eps import FiniteFieldElecEps as EpsCalc
from toolbox.utils import *

atoms = io.read("coord.xyz")
task = EpsCalc(atoms, work_dir=".", efield_seq=[-0.2, 0.0, 0.2])
task.workflow()