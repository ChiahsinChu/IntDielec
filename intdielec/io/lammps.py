from ase import io, Atoms
import numpy as np


def get_energies(energy_file="energy.log"):
    energies = np.loadtxt(energy_file)[:, 2]
    return energies


def read_dump(dump_file="dump.lammpstrj"):
    traj = io.read(dump_file, index=":")
    coords = []
    forces = []
    boxs = []
    for atoms in traj:
        forces.append(atoms.get_forces())
        coords.append(atoms.get_positions())
        boxs.append(atoms.get_cell())
    coords = np.reshape(coords, (len(traj), -1))
    forces = np.reshape(forces, (len(traj), -1))
    boxs = np.reshape(boxs, (len(traj), -1))
    type_list = atoms.get_array('numbers')
    type_list = np.array(type_list) - 1
    return coords, forces, boxs, type_list


def write_dump(traj,
               type_map,
               start=0,
               step=1,
               out_file="out.lammpstrj",
               append=False):
    if isinstance(type_map, list):
        atype_dict = {}
        for ii, atype in enumerate(type_map, start=1):
            atype_dict[atype] = {}
            atype_dict[atype]["type"] = ii
            atype_dict[atype]["element"] = atype
    elif isinstance(type_map, dict):
        atype_dict = type_map
    else:
        raise AttributeError("Unknown type of type_map")

    if isinstance(traj, Atoms):
        _write_dump(traj, atype_dict, ts, out_file, append)
    else:
        nframe = len(traj)
        _ts = np.arange(start, step * nframe, step)
        for ts, atoms in zip(_ts, traj):
            _write_dump(atoms, atype_dict, ts, out_file, append=True)


def _write_dump(atoms, atype_dict, ts, out_file, append):
    if append:
        with open(out_file, "a", encoding='utf-8') as f:
            header = make_dump_header(atoms, ts)
            f.write(header)
            body = make_dump_body(atoms, atype_dict)
            f.write(body)
    else:
        with open(out_file, "w", encoding='utf-8') as f:
            header = make_dump_header(atoms, ts)
            f.write(header)
            body = make_dump_body(atoms, atype_dict)
            f.write(body)


def make_dump_header(atoms, ts):
    cell = atoms.cell.cellpar()
    nat = len(atoms)
    s = "ITEM: TIMESTEP\n%d\n" % ts
    s += "ITEM: NUMBER OF ATOMS\n"
    s += "%d\n" % nat
    s += "ITEM: BOX BOUNDS pp pp pp\n"
    s += "%.4f %.4f\n%.4f %.4f\n%.4f %.4f\n" % (0.0, cell[0], 0.0, cell[1],
                                                0.0, cell[2])
    if len(atoms.get_initial_charges()) > 0:
        s += "ITEM: ATOMS id type element x y z q\n"
    else:
        s += "ITEM: ATOMS id type element x y z\n"
    return s


def make_dump_body(atoms, atype_dict):
    if len(atoms.get_initial_charges()) > 0:
        q_flag = True
        charges = atoms.get_initial_charges()
    else:
        q_flag = False
    ps = atoms.get_positions()

    s = ""
    for atom in atoms:
        ii = atom.index
        if q_flag:
            s += "%d %d %s %.6f %.6f %.6f %.6f\n" % (
                ii + 1, atype_dict[atom.symbol]["type"],
                atype_dict[atom.symbol]["element"], ps[ii][0], ps[ii][1],
                ps[ii][2], charges[ii])
        else:
            s += "%d %d %s %.6f %.6f %.6f\n" % (
                ii + 1, atype_dict[atom.symbol]["type"],
                atype_dict[atom.symbol]["element"], ps[ii][0], ps[ii][1],
                ps[ii][2])
    return s
