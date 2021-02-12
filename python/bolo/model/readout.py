
from collections import OrderedDict as odict

import numpy as np

from bolo.model.dummy import Property, Model

from bolo.ctrl.utils import is_not_none
from bolo.calc import physics

class Readout(Model):
    """
    Foreground object contains the foreground parameters for the sky
    """
    squid_nei = Property(dtype=float)
    read_noise_frac = Property(dtype=float, default=0.1)
    dwell_time = Property(dtype=float)
    revisit_rate = Property(dtype=float)
    nyquist_inductance = Property(dtype=float)

