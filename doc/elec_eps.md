# electronic dielectric constant

With the workflow in this package, you will be able to finish the calculation of electronic dielectric constant with a single job submission.

Example for bash code to submit the job (with a specific conda virtual environment `env_name`):

```bash
# set env var for this package and cp2k
module load miniconda/3
source activate env_name
module load mkl/latest mpi/latest gcc/9.3.0
module load cp2k/9.1

python run.py 1>eps_cal.stdout 2>eps_cal.stderr
```

> Environment variables can be loaded either in the bash (above) or in the json (see following).

## `ElecEps`

With this class, you can specify the potential drop crossing the cell.

```python
import numpy as np
from intdielec.workflow.elec_eps import ElecEps

task = ElecEps(work_dir="eps_cal")
task.set_v_seq(np.linspace(-1.0, 1.0, 3))
task.workflow(configs="param.json")
# task.make_plots(out=["lin_test", "pdos"], figure_name_suffix="_new")
```

Example for `param.json`:

```json
{
  "load_module": [],
  "command": "mpiexec.hydra cp2k.popt",
  "preset": {
    "pos_dielec": [5, 55]
  },
  "calculate": {
    "pos_vac": 7
  }
}
```

If you have finished the workflow and just want to export some figures, you can use:

```python
from intdielec.workflow.elec_eps import ElecEps

# calculations in `eps_cal` have been finished
task = ElecEps(work_dir="eps_cal")
task.make_plots(out=["inveps", "lin_test"], figure_name_suffix="_eps")
task.make_plots(out="pdos", figure_name_suffix="_pdos")
```

## `IterElecEps`

With this class, you can find a bias potential to reproduce the PBC Hartree potential in the bulk water region.

```python
from intdielec.workflow.elec_eps import IterElecEps

task = IterElecEps(work_dir="eps_cal")
task.workflow(configs="param.json")
```

Example for `param.json`:

```json
{
  "load_module": [],
  "command": "mpiexec.hydra cp2k.popt",
  "max_loop": 20,
  "pbc_preset": {
    "eps_scf": 1e-6
  },
  "ref_preset": {
    "eps_scf": 1e-4
  },
  "search_preset": {
    "eps_scf": 1e-5
  },
  "preset": {
    "eps_scf": 1e-6
  }
}
```

## `DualIterElecEps`

With this class, you can 1) find a bias potential to reproduce the PBC Hartree potential in the bulk water region and 2) prevent level alignment issue.

```python
from intdielec.workflow.elec_eps import DualIterElecEps

task = DualIterElecEps(work_dir="eps_cal")
task.workflow(configs="param.json")
```

Example for `param.json`: the same as that in [`IterElecEps`](#itereleceps).

## `QMMMIterElecEps`

With this class, you can 1) find a bias potential to reproduce the PBC Hartree potential in the bulk water region and 2) add a MM water layer out of the QM water layer (trying to stablise the QM water).

```python
from intdielec.workflow.elec_eps import QMMMIterElecEps

task = QMMMIterElecEps(work_dir="eps_cal")
task.workflow(configs="param.json")
```

Example for `param.json`:

```json
{
  "load_module": [],
  "command": "mpiexec.hydra cp2k.popt",
  "max_loop": 20,
  "l_qm_wat": 13,
  "l_mm_wat": 13,
  "pbc_preset": {
    "eps_scf": 1e-6
  },
  "ref_preset": {
    "eps_scf": 1e-4
  },
  "search_preset": {
    "eps_scf": 1e-5
  },
  "preset": {
    "eps_scf": 1e-6
  }
}
```

## manual version

Here is a simple example, in which you need to submit all DFT calculation by yourself:

```python
from intdielec.workflow.elec_eps import ElecEps

task = ElecEps(atoms, work_dir="eps_cal")
task.ref_preset()
# submit job and wait for calculation finish...
task.ref_calculate(vac_region=[45, 10.])
task.set_v_seq(np.arange(-3.5, 3.6, 0.5))
task.preset(pos_dielec=[5., 55.])
# submit and wait for calculation finish...
task.calculate(pos_vac=8.0)
task.make_plots()
```

methods and keywords... (TBC)
