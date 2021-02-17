
from collections import OrderedDict as odict

import numpy as np

from cfgmdl import Choice, Property, Model, Parameter

from bolo.ctrl.utils import is_not_none
from bolo.calc import physics

class Readout(Model):
    """
    Instrument readout model
    """
    squid_nei = Property(dtype=float)
    read_noise_frac = Parameter(default=0.1)
    dwell_time = Property(dtype=float)
    revisit_rate = Property(dtype=float)
    nyquist_inductance = Property(dtype=float)
    rtype = Choice(choices=["a", "b", "c"], default="a")
