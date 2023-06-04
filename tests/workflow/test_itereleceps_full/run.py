from ase import io

from intdielec.workflow.elec_eps import IterElecEps


work_dir = "."
atoms = io.read("../data/coord.xyz")

task = IterElecEps(work_dir=work_dir, atoms=atoms)
task.workflow(configs="param.json")
