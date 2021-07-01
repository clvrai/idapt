"""genal util function"""
import numpy as np
import os
import subprocess as sp
import torch
from collections import OrderedDict
from gym.spaces.box import Box
from gym.spaces.dict import Dict


def get_gpu_memory():
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def dict_add_prefix(x, prefix):
    "add prefix to all keys in @x which is a dict"
    out = {}
    for k, v in x.items():
        out[prefix + "_" + k] = v
    return out


def cat_dict(x):
    x = list(x.values())
    out = [e.unsqueeze(0) for e in x]
    return torch.cat(out, dim=-1)


def join_spaces(s1, s2):
    """join two input Dict spaces, ob + ac or ob + ob"""
    assert isinstance(s1, Dict) and isinstance(s2, Dict)
    assert len(s1.spaces.keys()) == 1 and len(s2.spaces.keys()) == 1

    k1 = "ob"
    k2 = "ob" if "ob" in s2.spaces.keys() else "ac"

    shape = (sum(x) for x in zip(s1.spaces[k1].shape, s2.spaces[k2].shape))
    high = np.concatenate((s1.spaces[k1].high, s2.spaces[k2].high))
    low = np.concatenate((s1.spaces[k1].low, s2.spaces[k2].low))
    return Dict({k1: Box(high=high, low=low)})


def join_dicts(d1, d2):
    """join ob/ac dicts"""
    if "ob" in d1.keys() and "ac" in d2.keys():
        joined_dict = {}
        joined_dict["ob"] = np.concatenate((d1["ob"], d2["ac"]), axis=-1)
        return joined_dict
    elif "ob" in d1.keys() and "ob" in d2.keys():
        joined_dict = {}
        joined_dict["ob"] = torch.cat((d1["ob"], d2["ob"]), dim=-1)
        return joined_dict
    else:
        raise NotImplementedError
