import numpy as np
import time
from ase import io
import glob
import os
import multiprocessing as mp

import MDAnalysis as mda
from MDAnalysis import transformations as trans

from zjxpack.postprocess.metal import ECMetal

from WatAnalysis.parallel import parallel_exec
from WatAnalysis.dielectric import ParallelInverseDielectricConstant as PIDC
# from WatAnalysis.dielectric import InverseDielectricConstant as IDC

dim = [11.246, 11.246, 35.94, 90, 90, 90]

_dnames = glob.glob("*")
dnames = []
for dname in _dnames:
    if os.path.isdir(dname):
        if os.path.exists(os.path.join(dname, "final_system.data")) and not os.path.exists(os.path.join(dname, "dielectric/inveps.npy")):
            dnames.append(dname)
dnames = np.sort(dnames)

atoms = io.read("/data/jxzhu/2022_leiden/02.nnp_validation/input/coord.xyz")
atoms.set_cell(dim)
atoms = ECMetal(atoms, metal_type="Pt", surf_atom_num=16)
surf_ids = atoms.get_surf_idx()
# print(surf_ids)

for dname in dnames:
    print("Start: ", dname)
    
    # load trajectory
    traj = os.path.join(dname, "dump.lammpstrj")
    u = mda.Universe("/data/jxzhu/2022_leiden/02.nnp_validation/input/interface.psf", 
                     traj, 
                     topology_format="PSF",
                     format="LAMMPSDUMP")
    #transform = trans.boxdimensions.set_dimensions(dim)
    #u.trajectory.add_transformations(transform)

    task = PIDC(
            universe=u,
            bins=np.arange(0, 10, 0.2),
            axis="z",
            temperature=330,
            make_whole=False,
            surf_ids=surf_ids,
            c_ag="name O",
            select_all=True,
            )
    parallel_exec(task.run, 0, 2000000, 1, 20)
    
    if not os.path.exists(os.path.join(dname, "dielectric")):
        os.makedirs(os.path.join(dname, "dielectric"))
    np.save(os.path.join(dname, "dielectric/inveps.npy"), task.results["inveps"])
    np.save(os.path.join(dname, "dielectric/M2.npy"), task.results["M2"])
    np.save(os.path.join(dname, "dielectric/m.npy"), task.results["m"])
    np.save(os.path.join(dname, "dielectric/mM.npy"), task.results["mM"])
    np.save(os.path.join(dname, "dielectric/M.npy"), task.results["M"])
    
    print("End: ", dname)


