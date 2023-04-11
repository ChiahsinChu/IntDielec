# IntDielec

## To do list

- [ ] benchmark of my code and MAICoS (and decide if I need to move to MAICoS!)
- [ ] exception capture class for all calculators
- [ ] write class for workflow plotting
- [ ] NNP validation code used in this work

## Introduction

## Installation

```bash
git clone --recursive https://github.com/ChiahsinChu/IntDielec.git .
cd IntDielec
pip install .
```

If you don't use `--recursive`, you will miss the submodule(s). You can, alternatively, download the submodule(s) in two steps:

```bash
git clone --recursive https://github.com/ChiahsinChu/IntDielec.git .
cd IntDielec
git submodule update --init --recursive
```

### write user setup in config file

You can set some user-defined parameters (e.g., directory for the basis set) in the `IntDielec/config.json`. Then, you don't need to define the variables when you call the relevant functions/methods.

Here is an example of the `config.json`:

```json
{
  "io": {
    "cp2k": {
      "input": {
        "project": "eps_cal",
        "pp_dir": "/home/user/basis",
        "cutoff": 800,
        "rel_cutoff": 50
      }
    }
  }
}
```

## User guide

### calculation of electronic dielectric constant

Here is a simple example:

```python
from intdielec.workflow.elec_eps import ElecEps

task = ElecEps(atoms, work_dir="eps_cal")
task.ref_preset()
# submit and wait for calculation finish...
task.ref_calculate(vac_region=[45, 10.])
task.set_v_seq(np.arange(-3.5, 3.6, 0.5))
task.preset(pos_dielec=[5., 55.])
# submit and wait for calculation finish...
task.calculate(pos_vac=8.0)
task.make_plots()
```

methods and keywords... (TBC)

### calculation of orientational dielectric constant

### workflow

Example for bash code to submit the job (with a specific conda virtual environment `env_name`):

```bash
module load miniconda/3
source activate env_name
module load mkl/latest mpi/latest gcc/9.3.0
module load cp2k/9.1

python run.py 1>eps_cal.stdout 2>eps_cal.stderr
```

> Environment variables can be loaded either in the bash or in the json.

- `ElecEps`

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

- `IterElecEps`

  Example for python code to run the workflow:

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

- `DualIterElecEps`

  Example for python code to run the workflow:

  ```python
  from intdielec.workflow.elec_eps import DualIterElecEps

  task = DualIterElecEps(work_dir="eps_cal")
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

- `QMMMIterElecEps`
  Example for python code to run the workflow:

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

## Developer guide

This package is constructed in the following way. TBC

## Reference
