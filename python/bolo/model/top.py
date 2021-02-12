
from collections import OrderedDict as odict

import numpy as np

from bolo.model.dummy import Property, Model
from bolo.model.foregrounds import Universe
from bolo.model.instrument import Instrument

from bolo.ctrl.utils import is_not_none
from bolo.calc import physics

class Top(Model):
    """
    Foreground object contains the foreground parameters for the sky
    """
    instrument = Property(dtype=Instrument)
    universe = Property(dtype=Universe)

    

