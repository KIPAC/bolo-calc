"""This module contains functions to help manage configuration for the
offline analysis of LSST Electrical-Optical testing"""

import sys
import os
import copy
import numpy as np

from collections import OrderedDict


def is_none(val):
    """Check to see if a value is none"""
    return val in [None, 'none', 'None', np.nan]

def is_not_none(val):
    """Check to see if a value is not none"""
    return val not in [None, 'none', 'None', np.nan]


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
