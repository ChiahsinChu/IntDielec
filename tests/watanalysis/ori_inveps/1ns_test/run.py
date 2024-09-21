# SPDX-License-Identifier: LGPL-3.0-or-later
import time

import MDAnalysis as mda
from toolbox.utils.utils import save_dict

from intdielec.utils import *
from intdielec.watanalysis.dielectric import InverseDielectricConstant as IDC

traj = "dump.lammpstrj"
topo = "interface.psf"
surf_ids = [
    [291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306],
    [371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386],
]

start_time = time.time()
u = mda.Universe(topo, traj, topology_format="PSF", format="LAMMPSDUMP")
end_time = time.time()
print("Used time (Load traj): %f [s]" % (end_time - start_time))

ag = u.select_atoms("resname R1")
# print(ag)

task_serial = IDC(
    atomgroups=ag,
    bin_edges=np.arange(0, 15.1, 0.1),
    temperature=330,
    img_plane=0.9,
    surf_ids=surf_ids,
)
start_time = time.time()
task_serial.run()
end_time = time.time()

print("Used time (Analyse): %f [s]" % (end_time - start_time))

print("nframe: %d" % task_serial.n_frames)

save_dict(task_serial.results, "result.pkl")
