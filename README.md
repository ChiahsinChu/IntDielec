# IntDielec

## To do list

- [ ] NNP validation code used in this work

## Introduction

## Installation

```bash
git clone --recursive https://github.com/ChiahsinChu/IntDielec.git ./IntDielec
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
fig, axs = task.plot()
```

methods and keywords... (TBC)

### calculation of orientational dielectric constant

### workflow

TBC

## Developer guide

This package is constructed in the following way. TBC

## Reference
