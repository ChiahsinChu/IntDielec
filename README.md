# IntDielec

## To do list

- [ ] write class for workflow plotting
- [ ] NNP validation code used in this work

## Introduction

This is python package for investigating dielectric constant distribution at metal/water interface. Some analysis classes are written on the basis of [MDAnalysis]() package (v2.3.0).
Submodule [`toolbox`](https://github.com/ChiahsinChu/toolbox) is used.

If you have any question/suggestion, please feel free to reach me via *jiaxinzhu@stu.xmu.edu.cn*.

## Installation

```bash
git clone --recursive https://github.com/ChiahsinChu/IntDielec.git .
cd IntDielec
pip install .
```

If you don't use `--recursive`, you will miss the submodule(s). You can, alternatively, download the submodule(s) in two steps:

```bash
git clone https://github.com/ChiahsinChu/IntDielec.git .
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

- [neural network potential](doc/nnp.md)
- [electronic electric constant](doc/elec_eps.md)
- [orientational electric constant](doc/orient_eps.md)

## Developer guide

This package is constructed in the following way. TBC

## Reference
