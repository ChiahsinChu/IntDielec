import MDAnalysis as mda
import numpy as np

from intdielec.watanalysis.acf import SelectedDipoleACF

# timestep = 10 * 0.5 = 5 [fs]
traj = "dump.xtc"
topo = "interface.psf"
u = mda.Universe(topo, traj, topology_format="PSF", format="XTC")
refs = [u.select_atoms("index 291:306"), u.select_atoms("index 371:386")]

task = SelectedDipoleACF(u.select_atoms("name O or name H"),
                         dts=np.arange(0, 2000, 10),
                         refs=refs,
                         cutoff=2.7)
task.run()
task.save("water_a_acf.txt")

task = SelectedDipoleACF(u.select_atoms("name O or name H"),
                         dts=np.arange(0, 2000, 10),
                         refs=refs,
                         cutoff=4.5)
task.run()
task.save("water_ab_acf.txt")