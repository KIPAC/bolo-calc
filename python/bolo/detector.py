
from collections import OrderedDict as odict

import numpy as np

from cfgmdl import Property, Model

from bolo.ctrl.utils import is_not_none
from bolo.calc import physics

class Detector(Model):
    """
    Detector Model
    """
    fractional_bandwidth = Property(dtype=float, default=.35)
    band = Property(dtype=float, default=150.E+9)
    pixel_size = Property(dtype=float, default=6.8E-3)
    num_det_per_water = Property(dtype=int, default=542)
    num_wafer_per_optics_tube = Property(dtype=int, default=1)
    num_optics_tube = Property(dtype=int, default=3)
    waist_factor = Property(dtype=int, default=3)
    det_eff = Property(dtype=float, default=1)
    psat = Property(dtype=float)
    carrier_index = Property(dtype=float, default=3)
    Tc = Property(dtype=float, default=.165)
    Tc_fraction = Property(dtype=float)
    G = Property(dtype=float)
    Flink = Property(dtype=float)
    Yield = Property(dtype=float)
    response_factor = Property(dtype=float)
    bolo_resistance = Property(dtype=float)

