
from collections import OrderedDict as odict

import numpy as np

from cfgmdl import Property, Model
from bolo.model.foregrounds import Universe
from bolo.model.instrument import Instrument

from bolo.ctrl.utils import is_not_none
from bolo.calc import physics

class Top(Model):
    """
    Top level model
    """
    instrument = Property(dtype=Instrument)
    universe = Property(dtype=Universe)

    

