
from collections import OrderedDict as odict

import numpy as np

from bolo.model.dummy import Property, Derived, Model

from bolo.model.camera import Camera
from bolo.model.detector import Detector
from bolo.model.readout import Readout
from bolo.model.opticalChain import Optics_Base, build_optics_class


from bolo.ctrl.utils import is_not_none
from bolo.calc import physics

class Instrument(Model):
    """
    Collection of instrument sub-system models
    """
    camera = Property(dtype=Camera, required=True)
    detector = Property(dtype=Detector, required=True)
    readout = Property(dtype=Readout, required=True)
    optics = Property(dtype=dict, required=True)
    optical_chain = Derived(dtype=Optics_Base)

    def _load_optical_chain(self):
        return build_optics_class(**self.optics)
        

    

