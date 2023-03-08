from ase import io
import numpy as np

from intdielec.workflow.elec_eps import ElecEps


work_dir = "."
atoms = io.read("../data/half_coord.xyz")

task = ElecEps(work_dir=work_dir, atoms=atoms)
task.set_v_seq(np.arange(-0.5, 0.6, 0.5))
task.workflow(configs="../data/param_v8.2.json")
