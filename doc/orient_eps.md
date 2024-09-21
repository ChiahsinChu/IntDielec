# orientational dielectric constant

## `InverseDielectricConstant`

With this class, you can calculate orientational dielectric constant via classical water model (e.g. SPC/E water).

```python
import MDAnalysis as mda
import numpy as np

from intdielec.watanalysis.dielectric import InverseDielectricConstant as IDC


topo = "interface.psf"
traj = "dump.lammpstrj"
surf_ids = [np.arange(0, 16), np.arange(16, 32)]

u = mda.Universe(topo, traj, topology_format="PSF", format="LAMMPSDUMP")
ag = u.select_atoms("name O or name H")
task = IDC(
    atomgroups=ag, bin_width=0.1, surf_ids=surf_ids, temperature=330, img_plane=0.935
)
task.run()
task.save()
```

## `AdInverseDielectricConstant`

Estimate the contribution of chemisorption.

```python
import MDAnalysis as mda
import numpy as np

from intdielec.watanalysis.dielectric import InverseDielectricConstant as IDC


topo = "interface.psf"
traj = "dump.lammpstrj"
surf_ids = [np.arange(0, 16), np.arange(16, 32)]

u = mda.Universe(topo, traj, topology_format="PSF", format="LAMMPSDUMP")
ag = u.select_atoms("name O or name H")
task = IDC(
    atomgroups=ag,
    bin_width=0.1,
    surf_ids=surf_ids,
    temperature=330,
    img_plane=0.935,
    cutoff=2.7,
    sfactor=2,
    calc_unscaled=True,
)
task.run()
task.save()
```

## Deep-Wannier model

With this class, you can calculate orientational dielectric constant via Deep Wannier model.

```python
import MDAnalysis as mda
import numpy as np

from intdielec.watanalysis.dielectric import DPInverseDielectricConstant as DPIDC


topo = "interface.psf"
traj = "dump.lammpstrj"
surf_ids = [np.arange(0, 16), np.arange(16, 32)]

u = mda.Universe(topo, traj, topology_format="PSF", format="LAMMPSDUMP")
ag = u.select_atoms("name O or name H")
task = IDC(
    atomgroups=ag,
    model="graph.pb",
    bin_width=0.1,
    surf_ids=surf_ids,
    temperature=330,
    img_plane=0.935,
)
task.run()
task.save()
```
