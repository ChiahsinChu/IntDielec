axis_dict = {"x": 0, "y": 1, "z": 2}


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
        else:
            old_d[k] = update_d[k]