from collections import OrderedDict as odict

import numpy as np

from cfgmdl import Property, Model
from bolo.ctrl.utils import expand_dict_from_defaults_and_elements


class OpticalElement(Model):
    """Model for a single optical element"""
    
    temperature = Property(dtype=float, required=True, format="%.2e")
    absorption = Property(dtype=list, required=True, format="%.2e")
    reflection = Property(dtype=list, required=True, format="%.2e")
    thickness = Property(dtype=float, format="%.2e")
    index = Property(dtype=float, format="%.2e")
    loss_tangent = Property(dtype=float, format="%.2e")
    conductivity = Property(dtype=float, format="%.2e")
    surface_rough = Property(dtype=float, format="%.2e")
    spillover = Property(dtype=float, format="%.2e")
    spillover_temp = Property(dtype=float, format="%.2e")
    scatter_frac = Property(dtype=float, format="%.2e")
    scatter_temp = Property(dtype=float, format="%.2e")


class Optics_Base(Model):
    pass


def build_optics_class(name="Optics", **kwargs):
    """ Build a class that consists of a set of OpticalElements

    Parameter
    ---------
    name : `str`
        The name of the new class
    kwargs : Heirachical dictionary used to build elements


    Returns
    -------
    optics_class : `type`
        The new class, which has all the requested properties

    """
    kwcopy = kwargs.copy()
    if 'default' in kwcopy:
        use_dict = expand_dict_from_defaults_and_elements(kwcopy.pop('default'), kwcopy.pop('elements'))
    else:
        use_dict = kwcopy

    properties = { key:Property(dtype=OpticalElement) for key in use_dict.keys() }
    optics_class = type(name, (Optics_Base, ), properties)
    return optics_class(**use_dict)


    
