# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np


def check_water(atoms, H_ids=None, O_ids=None, OH_cutoff=1.3):
    """
    Parameters
    ----------
    atoms : ASE Atoms object
        PBC should be pre-set
    H_ids/O_ids : list or array
        indices of H/O in water
    OH_cutoff : float (1.3)
        cutoff to define O-H bond
    """
    full_type = np.array(atoms.get_chemical_symbols())
    if H_ids is None:
        H_ids = np.arange(len(atoms))[full_type == "H"]
    if O_ids is None:
        O_ids = np.arange(len(atoms))[full_type == "O"]

    atoms.set_pbc(True)
    OH_ids = []
    H3O_ids = []
    # loop every water
    for ii in O_ids:
        ds = atoms.get_distances(ii, H_ids, mic=True)
        n_H = len(np.zeros_like(ds)[ds < OH_cutoff])
        if n_H == 1:
            OH_ids.append(ii)
        if n_H == 3:
            H3O_ids.append(ii)
        if len(OH_ids) > 0 or len(H3O_ids) > 0:
            print("WARNING!")


# def convert(
#     atoms,
#     metal_type="Pt",
#     n_surf=16,
#     l_water=[10., 16.],
# ):
#     """
#     OHHOHH...MM...
#     electrode-electrolyte-electrode
#     """
#     atoms.set_pbc(True)

#     coords = atoms.get_positions()
#     cell = atoms.get_cell()
#     spec = np.array(atoms.get_chemical_symbols())

#     atoms.set_positions(coords - np.array([0., 0., cell[2][2]]))
#     atoms.wrap()
#     z = atoms.get_positions()[:, 2]
#     z_hi = np.sort(z[spec == "Pt"])[-n_surf:].mean()
#     z_lo = np.sort(z[spec == "Pt"])[:n_surf].mean()

#     atoms.set_positions(coords - np.array([0., 0., z_lo - 10.]))
#     atoms.wrap()
#     z_hi = z_hi - z_lo + 10.

#     mask_O = (spec == "O")
#     mask_z = (z <= z_hi)
#     mask = mask_O & mask_z
#     full_ids = np.arange(len(atoms))
#     O_ids = full_ids[mask]
#     H_ids = np.append(O_ids + 1, O_ids + 2)
#     water_ids = np.append(O_ids, H_ids)
#     pt_ids = full_ids[spec == "Pt"]
#     sel_ids = np.sort(np.append(water_ids, pt_ids))
#     new_atoms = atoms[sel_ids]
#     new_atoms.positions += np.array([0., 0., 10.])
#     new_atoms.wrap()
#     new_atoms.positions += np.array([0., 0., 10.])
#     new_atoms.set_cell(cell)

#     return (atoms_lo, atoms_hi)

# def convert_lo(atoms):
#     p = atoms.get_positions()
#     z = p[:, 2]
#     spec = np.array(atoms.get_chemical_symbols())
#     mask_O = (spec == "O")
#     mask_z = (z <= 21)
#     mask = mask_O & mask_z
#     full_ids = np.arange(len(atoms))
#     O_ids = full_ids[mask]
#     H_ids = np.append(O_ids + 1, O_ids + 2)
#     water_ids = np.append(O_ids, H_ids)
#     pt_ids = full_ids[spec == "Pt"]
#     sel_ids = np.sort(np.append(water_ids, pt_ids))
#     new_atoms = atoms[sel_ids]
#     new_atoms.positions += np.array([0., 0., 10.])
#     new_atoms.wrap()
#     new_atoms.positions += np.array([0., 0., 10.])
#     new_atoms.set_cell(cell)
#     return new_atoms

# def convert_hi(atoms):
#     p = atoms.get_positions()
#     # inverse cell
#     p = -p + np.array([0., 0., 35.94])
#     atoms.set_positions(p)
#     z = p[:, 2]
#     spec = np.array(atoms.get_chemical_symbols())
#     mask_O = (spec == "O")
#     mask_z = (z <= 21)
#     mask = mask_O & mask_z
#     full_ids = np.arange(len(atoms))
#     O_ids = full_ids[mask]
#     H_ids = np.append(O_ids + 1, O_ids + 2)
#     water_ids = np.append(O_ids, H_ids)
#     pt_ids = full_ids[spec == "Pt"]
#     sel_ids = np.sort(np.append(water_ids, pt_ids))
#     new_atoms = atoms[sel_ids]
#     new_atoms.positions += np.array([0., 0., 10.])
#     new_atoms.wrap()
#     new_atoms.positions += np.array([0., 0., 10.])
#     new_atoms.set_cell(cell)
#     return new_atoms
