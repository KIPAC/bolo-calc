
from collections import OrderedDict as odict

import numpy as np

from bolo.model.dummy import Property, Model

from bolo.ctrl.utils import is_not_none
from bolo.calc import physics

class Camera(Model):
    """
    Foreground object contains the foreground parameters for the sky
    """
    boresite_elevation = Property(dtype=float, default=0., format="%.2e")
    optical_coupling = Property(dtype=float, default=1., format="%.2e")
    f_number = Property(dtype=float, default=2.5, format="%.2e")
    bath_temperature = Property(dtype=float, default=0.1, format="%.2e")

