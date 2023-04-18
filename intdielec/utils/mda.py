import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import translate, wrap


def make_selection(
    sel_region,
    surf_ids,
    c_ag="name O",
    select_all=False,
    bonded=False,
):
    """
    sel_region:
        selected region
    """
    assert len(sel_region) == 2
    assert len(surf_ids) == 2

    surf_lo = make_index_selection(surf_ids[0])
    surf_lo = "(" + surf_lo + ")"
    surf_hi = make_index_selection(surf_ids[1])
    surf_hi = "(" + surf_hi + ")"

    sel_region = np.abs(sel_region)
    lower_bound = np.min(sel_region)
    upper_bound = np.max(sel_region)

    surf_lo_region = make_relporp_selection(surf_lo,
                                            [lower_bound, upper_bound])
    surf_hi_region = make_relporp_selection(surf_hi,
                                            [-upper_bound, -lower_bound])
    select = "(" + surf_lo_region + ") or (" + surf_hi_region + ")"

    if c_ag is not None:
        select = "(" + select + ") and (%s)" % c_ag

    if select_all:
        select = "byres (" + select + ")"

    if bonded:
        select = "bonded (" + select + ")"

    return select


def make_index_selection(id_list):
    selection = "(index %d)" % id_list[0]
    for ii in id_list[1:]:
        selection = selection + " or (index %d)" % ii
    return selection


def make_relporp_selection(sel_ag, sel_region, direction="z"):
    """
    lower_bound < sel <= upper_bound
    """
    lower_bound = sel_region[0]
    upper_bound = sel_region[1]

    lo_region = "relprop %s > %f " % (direction, lower_bound) + sel_ag
    lo_region = "(" + lo_region + ")"
    hi_region = "relprop %s <= %f " % (direction, upper_bound) + sel_ag
    hi_region = "(" + hi_region + ")"

    selection = lo_region + " and " + hi_region
    return selection


def make_selection_two(
    sel_region,
    surf_ids,
    c_ag=None,
    select_all=False,
    bonded=False,
):
    """
    sel_region:
        selected region
    """
    assert len(sel_region) == 2
    assert len(surf_ids) == 2

    surf_lo = make_index_selection(surf_ids[0])
    surf_lo = "(" + surf_lo + ")"
    surf_hi = make_index_selection(surf_ids[1])
    surf_hi = "(" + surf_hi + ")"

    sel_region = np.abs(sel_region)
    lower_bound = np.min(sel_region)
    upper_bound = np.max(sel_region)

    surf_lo_region = make_relporp_selection(surf_lo,
                                            [lower_bound, upper_bound])
    surf_hi_region = make_relporp_selection(surf_hi,
                                            [-upper_bound, -lower_bound])
    select = [surf_lo_region, surf_hi_region]

    if c_ag is not None:
        select[0] = "(" + select[0] + ") and (%s)" % c_ag
        select[1] = "(" + select[1] + ") and (%s)" % c_ag

    if select_all:
        select[0] = "byres (" + select[0] + ")"
        select[1] = "byres (" + select[1] + ")"

    if bonded:
        select[0] = "bonded (" + select[0] + ")"
        select[1] = "bonded (" + select[1] + ")"

    return select


def xyz_writer(u, out, start=0, end=None, step=1):
    if end is None:
        end = u.trajectory.n_frames

    if isinstance(out, str):
        with mda.Writer(out) as W:
            for ts in u.trajectory[start:end:step]:
                W.write(u)
    elif isinstance(out, list):
        assert len(out) == len(np.arange(start, end, step))
        for ts, fname in zip(u.trajectory[start:end:step], out):
            f = mda.Writer(fname)
            f.write(u)
    else:
        raise AttributeError("TBC")


def get_com_pos(universe, elem_type):
    elem_ag = universe.select_atoms("name " + elem_type)
    p = elem_ag.positions.copy()
    ti_com_z = p.mean(axis=0)[-1]
    return ti_com_z


def center(universe, elem_type, cell=None):
    """
    TBC
    """
    # check/set dimensions of universe
    if universe.dimensions is None:
        if cell is None:
            raise AttributeError('Cell parameter required!')
        else:
            universe.dimensions = cell
    # translate and wrap
    metal_com_z = get_com_pos(universe, elem_type)
    workflow = [translate([0, 0, -metal_com_z]), wrap(universe.atoms)]
    universe.trajectory.add_transformations(*workflow)
