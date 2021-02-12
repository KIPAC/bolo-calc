from collections import OrderedDict as odict

import numpy as np

from bolo.model.dummy import Property, Model
from bolo.ctrl.utils import expand_dict_from_defaults_and_elements


class OpticalElement(Model):
    temperature = Property(dtype=float, required=True, format="%.2e", unit='K')
    absorption = Property(dtype=list, required=True, format="%.2e")
    reflection = Property(dtype=list, required=True, format="%.2e")
    thickness = Property(dtype=float, format="%.2e", unit='mm')
    index = Property(dtype=float, format="%.2e")
    loss_tangent = Property(dtype=float, format="%.2e")
    conductivity = Property(dtype=float, format="%.2e")
    surface_rough = Property(dtype=float, format="%.2e")
    spillover = Property(dtype=float, format="%.2e")
    spillover_temp = Property(dtype=float, format="%.2e", unit='K')
    scatter_frac = Property(dtype=float, format="%.2e")
    scatter_temp = Property(dtype=float, format="%.2e", unit='K')

    
class Optics_Base(Model):
    pass


def build_optics_class(name="Optics", **kwargs):
    """ C'tor.  Build from a set of keyword arguments.
    """
    kwcopy = kwargs.copy()
    if 'default' in kwcopy:
        use_dict = expand_dict_from_defaults_and_elements(kwcopy.pop('default'), kwcopy.pop('elements'))
    else:
        use_dict = kwcopy

    properties = { key:Property(dtype=OpticalElement) for key in use_dict.keys() }
    optics_class = type(name, (Optics_Base, ), properties)
    return optics_class(**use_dict)


    
