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
task.ref_calculate(vac_region=[45, 10.0])
task.set_v_seq(np.arange(-3.5, 3.6, 0.5))
task.preset(pos_dielec=[5.0, 55.0])
# submit and wait for calculation finish...
task.calculate(pos_vac=8.0)
task.make_plots()
```

## Keywords in setup json files

| keyword           | type  | method            | description                                                                     | example                                                                              |
| ----------------- | ----- | ----------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `load_module`     | List  | all               | module to load                                                                  | `["intel/17.5.239", "mpi/intel/2017.5.239", "gcc/5.5.0", "cp2k/7.1"]`                |
| `command`         | str   | all               | command to run cp2k                                                             | `mpiexec.hydra cp2k_shell.popt`                                                      |
| `machine`         | Dict  | all               | machine setup for DPDispatcher                                                  | [here](https://docs.deepmodeling.com/projects/dpdispatcher/en/latest/machine.html)   |
| `resources`       | Dict  | all               | resources setup for DPDispatcher                                                | [here](https://docs.deepmodeling.com/projects/dpdispatcher/en/latest/resources.html) |
| `task`            | Dict  | all               | task setup for DPDispatcher                                                     | [here](https://docs.deepmodeling.com/projects/dpdispatcher/en/latest/task.html)      |
| `ref_preset`      | Dict  | all               | DFT setup for reference calculation (dipole correction)                         | see [below](#dft-setup)                                                              |
| `ref_calculate`   | Dict  | all               | post-processing setup for reference calculation (dipole correction)             | see [below](#post-processing-setup)                                                  |
| `preset`          | Dict  | all               | DFT setup for calculation with Dirichlet boundary conditions                    | see [below](#dft-setup)                                                              |
| `calculate`       | Dict  | all               | post-processing setup for calculation with Dirichlet boundary conditions        | see [below](#post-processing-setup)                                                  |
| `l_qm_wat`        | float | `IterElecEps`     | length of water for QM calculation                                              | `15.0`                                                                               |
| `l_wat_hartree`   | float | `IterElecEps`     | length of water layer for Hartree reference                                     | `10.0`                                                                               |
| `l_vac`           | float | `IterElecEps`     | length of vacuum region                                                         | `20.0`                                                                               |
| `pbc_preset`      | Dict  | `IterElecEps`     | DFT setup for PBC calculation calculation                                       | see [below](#dft-setup)                                                              |
| `search_preset`   | Dict  | `IterElecEps`     | DFT setup for searching calculation                                             | see [below](#dft-setup)                                                              |
| `convergence`     | float | `IterElecEps`     | convergence threshold for Hartree potential                                     | `1e-2`                                                                               |
| `max_loop`        | int   | `IterElecEps`     | maximum loop number for iterative calculation                                   | `20`                                                                                 |
| `guess_method`    | str   | `IterElecEps`     | guess method for searching                                                      | `ols_cut`                                                                            |
| `guess_setup`     | Dict  | `IterElecEps`     | guess setup for searching                                                       | see [below](#guess-setup)                                                            |
| `l_mm_wat`        | float | `QMMMIterElecEps` | length of water for MM calculation calculation                                  | `10.0`                                                                               |
| `max_loop_eps`    | int   | `DualIterElecEps` | maximum loop number for iterative calculation of electronic dielectric constant | `20`                                                                                 |
| `convergence_eps` | float | `DualIterElecEps` | convergence threshold for electronic dielectric constant                        | `5e-3`                                                                               |

### DFT setup

| keyword  | type | method | description                    | example |
| -------- | ---- | ------ | ------------------------------ | ------- |
| `dname`  | str  | all    | name of working directory      | `ref`   |
| `fparam` | Dict | all    | `fparam` for `Cp2kInput.write` |         |

- and other kwargs for `Cp2kInput`

### post-processing setup

| keyword      | type        | method    | description                            | example      |
| ------------ | ----------- | --------- | -------------------------------------- | ------------ |
| `vac_region` | List[float] | `ElecEps` | two points to calculate potential drop | `[45, 10.0]` |
| `pos_vac`    | float       | `ElecEps` | vaccum position for E-field referenc   | `10.0 `      |
| `save_fname` | str         | `ElecEps` | file name for output data              | `eps_data`   |

### guess setup

If `guess_method` is `ols_cut` (ordinary least squares) or `wls_cut` (weighted least squares), `nstep` is required to specify the number of maximal steps for the guess.

## Output

```bash
work_dir
├── eps_data_hi.pkl # data for upper surface
├── eps_data_lo.pkl # data for lower surface
├── figures
│   ├── eps_data_hi.png
│   └── eps_data_lo.png
├── pbc # directory for PBC calculation
├── ref_hi # directory for reference calculation (upper surface)
├── ref_lo # directory for reference calculation (lower surface)
├── search_hi.000000 # directory for searching matching Hartree potential
├── search_hi.000001
├── ...
├── search_lo.000000
├── search_lo.000001
├── ...
├── task_lo.000000 # directory for searching converged dielectric profile
├── task_lo.000001
├── ...
├── task_hi.000000
├── task_hi.000001
└── ...
```

Data in `eps_data_*.pkl` can be read as python dictionary `d`:

```python
print(d.keys())
# 'v', 'efield', 'v_grid', 'hartree', 'efield_vac', 'rho', 'v_prime', 'rho_pol', 'delta_efield_vac', 'delta_efield', 'v_prime_grid', 'delta_pol', 'inveps', 'lin_test'
```

- `v`: potential drop set in DBC calculation
- `efield`: electric profile
- `v_grid`: grid
- `hartree`: Hartree potential
- `efield_vac`: electric profile in vacuum
- `rho`: charge density profile

- `v_prime`: difference of potential drop
- `rho_pol`: polarisation charge density profile
- `delta_efield_vac`: difference of electric profile in vacuum
- `delta_efield`: difference of electric profile
- `v_prime_grid`: grid for difference quantity
- `delta_pol`: difference of polarisation profile
- `inveps`: inverse dielectric constant
- `lin_test`: standard deviation of `inveps` for different calculations
