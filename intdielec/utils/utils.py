import csv
import json
import os
import pickle
import h5py
import numpy as np


def iterdict(input_dict, out_list, loop_idx):
    """ 
    recursively generate a list of strings for further 
    print out CP2K input file

    Args:
        input_dict: dictionary for CP2K input parameters
        out_list: list of strings for printing
        loop_idx: record of loop levels in recursion
    Return:
        out_list
    """
    if len(out_list) == 0:
        out_list.append("\n")
    start_idx = len(out_list) - loop_idx - 2
    for k, v in input_dict.items():
        k = str(k)  # cast key into string
        #if value is dictionary
        if isinstance(v, dict):
            out_list.insert(-1 - loop_idx, "&" + k)
            out_list.insert(-1 - loop_idx, "&END " + k)
            iterdict(v, out_list, loop_idx + 1)
        # if value is list
        elif isinstance(v, list):
            if isinstance(v[0], dict):
                for _v in v:
                    out_list.insert(-1 - loop_idx, "&" + k)
                    out_list.insert(-1 - loop_idx, "&END " + k)
                    iterdict(_v, out_list, loop_idx + 1)
                #print(loop_idx)
                #print(input_dict)
                #print(out_list)
            else:
                for _v in v:
                    _v = str(_v)
                    out_list.insert(-1 - loop_idx, k + " " + _v)
        # if value is other type, e.g., int/float/str
        else:
            v = str(v)
            if k == "_":
                out_list[start_idx] = out_list[start_idx] + " " + v
            else:
                out_list.insert(-1 - loop_idx, k + " " + v)
                #out_list.insert(-1-loop_idx, v)
    return out_list


"""
def iterdict(input_dict, out_list=["\n"], loop_idx=0):
    if len(out_list) == 0:
        out_list.append("\n")
    start_idx = len(out_list) - loop_idx - 2
    n_repeat = -1
    for k,v in input_dict.items():
        k=str(k) # cast key into string
        #if value is dictionary
        if isinstance(v, dict):
            out_list.insert(-1-loop_idx, "&"+k)
            out_list.insert(-1-loop_idx, "&END "+k)
            iterdict(v, out_list, loop_idx+1)
        elif isinstance(v, list):
            n_repeat = len(v)
            #print(loop_idx)
            #print(input_dict)
            #print(out_list)
            #print(n_repeat)
            break
        else:
            v = str(v)
            if k == "_":
                out_list[start_idx] = out_list[start_idx] + " " + v
            else: 
                out_list.insert(-1-loop_idx, k+" "+v)
                #out_list.insert(-1-loop_idx, v)
    if n_repeat >= 0 :
        end_str = out_list[-1-loop_idx]
        del out_list[-1-loop_idx]
        start_str = out_list[-1-loop_idx]
        del out_list[-1-loop_idx]
        for i in range(n_repeat):
            tmp_dict = {}
            for k, v in input_dict.items():
                k=str(k)
                tmp_dict[k] = v[i]
            out_list.insert(-loop_idx, start_str)
            out_list.insert(-loop_idx, end_str)
            iterdict(tmp_dict, out_list, loop_idx)
    return out_list
"""


def update_dict(old_d, update_d):
    """
    source: dpgen.generator.lib.cp2k

    a method to recursive update dict
    :old_d: old dictionary
    :update_d: some update value written in dictionary form
    """
    import collections
    for k, v in update_d.items():
        if (k in old_d and isinstance(old_d[k], dict)
                and isinstance(update_d[k], collections.Mapping)):
            update_dict(old_d[k], update_d[k])
        # elif (k in old_d and isinstance(old_d[k], list)
        #       and isinstance(update_d[k], list)):
        #     old_d[k].extend(update_d[k])
        else:
            old_d[k] = update_d[k]


def symlink(src, _dst):
    dst = os.path.abspath(_dst)
    os.symlink(src, dst)


def save_dict(d, fname, format=None):
    if format is None:
        format = os.path.splitext(fname)[1][1:]
    try:
        globals()["save_dict_%s" % format](d, fname)
    except:
        raise AttributeError("Unknown format %s" % format)


def save_dict_json(d, fname):
    with open(fname, "w") as f:
        json.dump(d, f, indent=4)


def save_dict_csv(d, fname):
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=d.keys())
        writer.writeheader()
        writer.writerow(d)


def save_dict_pkl(d, fname):
    with open(fname, "wb") as f:
        pickle.dump(d, f)


def save_dict_hdf5(d, fname):
    with h5py.File(fname, "a") as f:
        n = len(f)
        dts = f.create_group("%02d" % n)
        for k, v in d.items():
            dts.create_dataset(k, data=v)


def load_dict(fname, format=None):
    if format is None:
        format = os.path.splitext(fname)[1][1:]
    try:

        return globals()["load_dict_%s" % format](fname)
    except:
        raise AttributeError("Unknown format %s" % format)


def load_dict_json(fname):
    with open(fname, "r") as f:
        d = json.load(f)
    return d


def load_dict_csv(fname):
    with open(fname, "r") as f:
        data = csv.reader(f)
        d = {rows[0]: rows[1] for rows in data}
    return d


def load_dict_pkl(fname):
    with open(fname, "rb") as f:
        d = pickle.load(f)
    return d


def load_dict_hdf5(fname):
    return {}


def get_efields(DeltaV, l: list, eps: list):
    r_field = 1. / np.array(eps)
    _delta_v = np.sum(np.array(l) * r_field)
    v_coeff = DeltaV / _delta_v
    return r_field * v_coeff