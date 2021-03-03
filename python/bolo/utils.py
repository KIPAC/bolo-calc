"""This module contains functions to help manage configuration for the
offline analysis of LSST Electrical-Optical testing"""

import os
import numpy as np

CONFIG_DIR = None

def is_none(val):
    """Check to see if a value is none"""
    return val in [None, 'none', 'None', np.nan]

def is_not_none(val):
    """Check to see if a value is not none"""
    return val not in [None, 'none', 'None', np.nan]


class CfgDir:
    """ Tiny class to find configuration files"""

    def __init__(self):
        """ Constructor """
        self.config_dir = None

    def set_dir(self, val):
        """ Set the top-level configuration directory"""
        self.config_dir = val

    def get_dir(self):
        """ Get the top-level configuration directory"""
        return self.config_dir

    def cfg_path(self, val):
        """ Build a path using the top-level configuration directory """
        return os.path.join(self.config_dir, val)

CFG_DIR = CfgDir()

set_config_dir = CFG_DIR.set_dir
get_config_dir = CFG_DIR.get_dir
cfg_path = CFG_DIR.cfg_path


def copy_dict(in_dict, def_dict):
    """Copy a set of key-value pairs to an new dict

    Parameters
    ----------
    in_dict : `dict`
        The dictionary with the input values
    def_dict : `dict`
        The dictionary with the default values

    Returns
    -------
    outdict : `dict`
        Dictionary with arguments selected from in_dict to overide def_dict
    """
    outdict = {key:in_dict.get(key, val) for key, val in def_dict.items()}
    return outdict


def pop_values(in_dict, keylist):
    """Pop a set of key-value pairs to an new dict

    Parameters
    ----------
    in_dict : `dict`
        The dictionary with the input values
    keylist : `list`
        The values to pop

    Returns
    -------
    outdict : `dict`
        Dictionary with only the arguments we have selected
    """
    outdict = {}
    for key in keylist:
        if key in in_dict:
            outdict[key] = in_dict.pop(key)
    return outdict



def update_dict_from_string(o_dict, key, val, subparser_dict=None):
    """Update a dictionary with sub-dictionaries

    Parameters
    ----------
    o_dict : dict
        The output

    key : `str`
        The string we are parsing

    val : `str`
        The value

    subparser_dict : `dict` or `None`
        The subparsers used to parser the command line

    """
    idx = key.find('.')
    use_key = key[0:idx]
    remain = key[idx+1:]
    if subparser_dict is not None:
        try:
            subparser = subparser_dict[use_key[1:]]
        except KeyError:
            subparser = None
    else:
        subparser = None

    if use_key not in o_dict:
        o_dict[use_key] = {}

    def_val = None
    if subparser is not None:
        def_val = subparser.get_default(remain)
    if def_val == val:
        return

    if remain.find('.') < 0:
        o_dict[use_key][remain] = val
    else:
        update_dict_from_string(o_dict[use_key], remain, val)



def expand_dict_from_defaults_and_elements(default_dict, elem_dict):
    """Expand a dictionary by copying defaults to a set of elements

    Parameters
    ----------
    default_dict : `dict`
        The defaults

    elem_dict : `dict`
        The elements

    Returns
    -------
    o_dict : `dict`
        The output dict
    """
    o_dict = {}
    for key, elem in elem_dict.items():
        o_dict[key] = default_dict.copy()
        if elem is None:
            continue
        o_dict[key].update(elem)
    return o_dict



def read_txt_to_np(fname):
    """ Read a txt file to a numpy array """
    ext = os.path.splitext(fname)[-1]
    if ext.lower() == '.txt':
        delim = None
    elif ext.lower() == '.csv':
        delim = ','
    else:
        raise ValueError("File %s is not csv or txt")
    return np.loadtxt(fname, unpack=True, dtype=np.float, delimiter=delim)


def reshape_array(val, shape):
    """ Reshape an array, but not a scalar

    This is useful for broadcasting many arrays to the same shape
    """
    if np.isscalar(val):
        return val
    return val.reshape(shape)
