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


"""
Functions for postprocessing of PartialHBAnalysis results
"""
import numpy as np
import os

from WatAnalysis.utils import get_cum_ave


def count_by_time(hbonds_result, start, stop, step=1, dt=1):
    """
    Adapted from MDA
    """
    indices, tmp_counts = np.unique(hbonds_result[:, 0],
                                    axis=0,
                                    return_counts=True)
    indices -= start
    indices /= step
    counts = np.zeros_like(np.arange(start, stop, step))
    counts[indices.astype(np.intp)] = tmp_counts
    return [np.arange(start, stop, step) * dt, counts, get_cum_ave(counts)]


def lifetime(hbonds_result,
             start,
             stop,
             step,
             dt,
             tau_max=20,
             window_step=1,
             intermittency=0):
    """
    Adapted from MDA
    """
    from MDAnalysis.lib.correlations import autocorrelation, correct_intermittency

    frames = np.arange(start, stop, step)
    found_hydrogen_bonds = [set() for _ in frames]
    for frame_index, frame in enumerate(frames):
        for hbond in hbonds_result[hbonds_result[:, 0] == frame]:
            found_hydrogen_bonds[frame_index].add(frozenset(hbond[2:4]))

    intermittent_hbonds = correct_intermittency(found_hydrogen_bonds,
                                                intermittency=intermittency)
    tau_timeseries, timeseries, timeseries_data = autocorrelation(
        intermittent_hbonds, tau_max, window_step=window_step)
    output = np.vstack([tau_timeseries, timeseries])
    output[0] = output[0] * dt
    return output


def fit_biexponential(tau_timeseries, ac_timeseries):
    """Fit a biexponential function to a hydrogen bond time autocorrelation function

    Return the two time constants

    Adapted from MDA
    """
    from scipy.optimize import curve_fit

    def model(t, A, tau1, B, tau2):
        """Fit data to a biexponential function.
        """
        return A * np.exp(-t / tau1) + B * np.exp(-t / tau2)

    params, params_covariance = curve_fit(model, tau_timeseries, ac_timeseries,
                                          [1, 0.5, 1, 2])

    fit_t = np.linspace(tau_timeseries[0], tau_timeseries[-1], 1000)
    fit_ac = model(fit_t, *params)

    return params, fit_t, fit_ac


def get_graphs(hbonds_result, output_dir):
    """
    Generate files for further analyses:

    n = total number of nodes
    m = total number of edges
    N = number of graphs
    * indices start from 1

    (1) A.txt (m lines)
        sparse (block diagonal) adjacency matrix for all graphs,
        each line corresponds to (row, col) resp. (node_id, node_id)

    (2) graph_indicator.txt (n lines)
        column vector of graph identifiers for all nodes of all graphs,
        the value in the i-th line is the graph_id of the node with node_id i

    (3) graph_labels.txt (N lines)
        class labels for all graphs in the dataset,
        the value in the i-th line is the class label of the graph with graph_id i
        1 1 1 1 ...

    (4) node_labels.txt (n lines)
        column vector of node labels,
        the value in the i-th line corresponds to the node with node_id i
        1 1 1 1 ... (maybe you can add the distance to the surface as the label?)

    Parameters
    ----------
    hbonds_result: (n, 9)-shape List
    output_dir:
        directory to save file
    """
    # number of edges (hydrogen bonds)
    n_edges = len(hbonds_result)
    graphs = np.zeros((n_edges, 2), dtype=np.int32)
    # get the range of every frames
    u, indices = np.unique(hbonds_result[:, 0], return_index=True)
    # number of frames
    n_frames = len(u)

    n_nodes = 0
    graph_id = 0
    for ii in range(n_frames - 1):
        # get the hbonds result of one frame
        start_id = indices[ii]
        end_id = indices[ii + 1]
        single_frame_hbonds = hbonds_result[start_id:end_id]
        # get the hbonds result upper/lower surfaces
        mask_lo, mask_hi = make_mask(single_frame_hbonds)
        d_a_pairs_lo = single_frame_hbonds[:, [1, 3]][mask_lo]
        d_a_pairs_hi = single_frame_hbonds[:, [1, 3]][mask_hi]

        # make graph for lower surfaces
        graph_lo, n_node_lo = make_graph(d_a_pairs_lo)
        # write graph_indicator (graph_id)
        graph_id = graph_id + 1
        if n_nodes == 0:
            graph_indicator = np.full((n_node_lo), graph_id, dtype=np.int32)
        else:
            graph_indicator = np.concatenate(
                (graph_indicator, np.full((n_node_lo),
                                          graph_id,
                                          dtype=np.int32)))
        # write graphs
        graph = graph_lo + n_nodes + 1
        np.copyto(graphs[start_id:start_id + len(graph)], graph)
        n_nodes = n_nodes + n_node_lo

        # make graph for upper surfaces
        graph_hi, n_node_hi = make_graph(d_a_pairs_hi)
        # write graph_indicator (graph_id)
        graph_id = graph_id + 1
        graph_indicator = np.concatenate(
            (graph_indicator, np.full((n_node_hi), graph_id, dtype=np.int32)))
        # write graphs
        graph = graph_hi + n_nodes + 1
        np.copyto(graphs[end_id - len(graph):end_id], graph)
        n_nodes = n_nodes + n_node_hi

    # the last frame
    start_id = end_id
    single_frame_hbonds = hbonds_result[start_id:]
    # get the hbonds result upper/lower surfaces
    mask_lo, mask_hi = make_mask(single_frame_hbonds)
    d_a_pairs_lo = single_frame_hbonds[:, [1, 3]][mask_lo]
    d_a_pairs_hi = single_frame_hbonds[:, [1, 3]][mask_hi]

    # make graph for lower surfaces
    graph_lo, n_node_lo = make_graph(d_a_pairs_lo)
    # write graph_indicator (graph_id)
    graph_id = graph_id + 1
    if n_nodes == 0:
        graph_indicator = np.full((n_node_lo), graph_id, dtype=np.int32)
    else:
        graph_indicator = np.concatenate(
            (graph_indicator, np.full((n_node_lo), graph_id, dtype=np.int32)))
    # write graphs
    graph = graph_lo + n_nodes + 1
    np.copyto(graphs[start_id:start_id + len(graph)], graph)
    n_nodes = n_nodes + n_node_lo

    # make graph for upper surfaces
    graph_hi, n_node_hi = make_graph(d_a_pairs_hi)
    # write graph_indicator (graph_id)
    graph_id = graph_id + 1
    graph_indicator = np.concatenate(
        (graph_indicator, np.full((n_node_hi), graph_id, dtype=np.int32)))
    # write graphs
    graph = graph_hi + n_nodes + 1
    np.copyto(graphs[-1 - len(graph):-1], graph)
    n_nodes = n_nodes + n_node_hi

    graph_labels = np.ones((2 * n_frames), dtype=np.int32)
    node_labels = np.ones_like(graph_indicator, dtype=np.int32)

    # save files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir, "A.txt"),
               graphs,
               delimiter=',',
               fmt="%d")
    np.savetxt(os.path.join(output_dir, "graph_indicator.txt"),
               graph_indicator,
               fmt="%d")
    np.savetxt(os.path.join(output_dir, "graph_labels.txt"),
               graph_labels,
               fmt="%d")
    np.savetxt(os.path.join(output_dir, "node_labels.txt"),
               node_labels,
               fmt="%d")


def make_graph(d_a_pairs):
    """
    Parameter
    ---------
    d_a_pairs: (n, 2)-shape List
        id of donors/acceptors in ONE graph (surface)
    
    Returns
    -------
    graph: (n, 2)-shape List
        adjacency matrix for input graph
    n_node: int
        number of nodes in the graph
    """
    u, indices = np.unique(d_a_pairs, return_inverse=True)
    n_node = len(u)
    graph = np.arange(n_node)[indices]
    graph = np.reshape(graph, (-1, 2))
    return graph, n_node


def make_mask(single_frame_hbonds):
    """
    make masks for the D-A pairs at upper/lower surface
    """
    mask_lo = (single_frame_hbonds[:, -1] > 0)
    mask_hi = (mask_lo == False)
    return mask_lo, mask_hi


def get_n_d_a_pairs(hbonds_result, donor_region=None, acceptor_region=None):
    """
    Get the number of specific D-A pairs
    e.g., donor(water A)-acceptor(water B)

    Parameters
    ----------
    hbonds_result: (n, 9)-shape List
    donor_region: (2, )-shape List
        z region for donor of interest
    acceptor_region: (2, )-shape List
        z region for acceptor of interest

    Return
    -------
    n_pairs: int
        total number of given D-A pairs
    """
    z_donor = np.abs(hbonds_result[:, -3])
    donor_mask = (z_donor >= donor_region[0]) & (z_donor < donor_region[1])
    z_acceptor = np.abs(hbonds_result[:, -1])
    acceptor_mask = (z_acceptor >= acceptor_region[0]) & (z_acceptor <
                                                          acceptor_region[1])
    mask = donor_mask & acceptor_mask
    n_pairs = len(np.zeros_like(z_donor)[mask])
    return n_pairs
