import glob

import numpy as np
from ase import io


def convert(
    atoms,
    metal_type="Pt",
    n_surf=16,
    l_water=[10., 16.],
):
    """
    OHHOHH...MM...
    electrode-electrolyte-electrode
    """
    atoms.set_pbc(True)

    coords = atoms.get_positions()
    cell = atoms.get_cell()
    spec = np.array(atoms.get_chemical_symbols())

    atoms.set_positions(coords - np.array([0., 0., cell[2][2]]))
    atoms.wrap()
    z = atoms.get_positions()[:, 2]
    z_hi = np.sort(z[spec == "Pt"])[-n_surf:].mean()
    z_lo = np.sort(z[spec == "Pt"])[:n_surf].mean()

    atoms.set_positions(coords - np.array([0., 0., z_lo - 10.]))
    atoms.wrap()
    z_hi = z_hi - z_lo + 10.

    mask_O = (spec == "O")
    mask_z = (z <= z_hi)
    mask = mask_O & mask_z
    full_ids = np.arange(len(atoms))
    O_ids = full_ids[mask]
    H_ids = np.append(O_ids + 1, O_ids + 2)
    water_ids = np.append(O_ids, H_ids)
    pt_ids = full_ids[spec == "Pt"]
    sel_ids = np.sort(np.append(water_ids, pt_ids))
    new_atoms = atoms[sel_ids]
    new_atoms.positions += np.array([0., 0., 10.])
    new_atoms.wrap()
    new_atoms.positions += np.array([0., 0., 10.])
    new_atoms.set_cell(cell)
    

    return (atoms_lo, atoms_hi)


def convert_lo(atoms):
    p = atoms.get_positions()
    z = p[:, 2]
    spec = np.array(atoms.get_chemical_symbols())
    mask_O = (spec == "O")
    mask_z = (z <= 21)
    mask = mask_O & mask_z
    full_ids = np.arange(len(atoms))
    O_ids = full_ids[mask]
    H_ids = np.append(O_ids + 1, O_ids + 2)
    water_ids = np.append(O_ids, H_ids)
    pt_ids = full_ids[spec == "Pt"]
    sel_ids = np.sort(np.append(water_ids, pt_ids))
    new_atoms = atoms[sel_ids]
    new_atoms.positions += np.array([0., 0., 10.])
    new_atoms.wrap()
    new_atoms.positions += np.array([0., 0., 10.])
    new_atoms.set_cell(cell)
    return new_atoms


def convert_hi(atoms):
    p = atoms.get_positions()
    # inverse cell
    p = -p + np.array([0., 0., 35.94])
    atoms.set_positions(p)
    z = p[:, 2]
    spec = np.array(atoms.get_chemical_symbols())
    mask_O = (spec == "O")
    mask_z = (z <= 21)
    mask = mask_O & mask_z
    full_ids = np.arange(len(atoms))
    O_ids = full_ids[mask]
    H_ids = np.append(O_ids + 1, O_ids + 2)
    water_ids = np.append(O_ids, H_ids)
    pt_ids = full_ids[spec == "Pt"]
    sel_ids = np.sort(np.append(water_ids, pt_ids))
    new_atoms = atoms[sel_ids]
    new_atoms.positions += np.array([0., 0., 10.])
    new_atoms.wrap()
    new_atoms.positions += np.array([0., 0., 10.])
    new_atoms.set_cell(cell)
    return new_atoms
