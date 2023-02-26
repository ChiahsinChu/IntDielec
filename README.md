# IntDielec

## To do list

- [ ] fix bug in import plot module
- [ ] set config files in installation
- [ ] merge the code in `WatAnalysis` for ori_eps calculation
- [ ] NNP validation code used in this work
- [ ] add bash file for task submission (temporary use)
- [ ] add workflow
- [ ] format to save data (SQL?)

## Introduction

## Installation

```bash
git clone https://github.com/ChiahsinChu/IntDielec.git ./IntDielec
cd IntDielec
pip install .
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

## Developer guide

This package is constructed in the following way. TBC

## Reference
