""" Instrument readout model """

from cfgmdl import Model
from .cfg import Variable

class Readout(Model):
    """
    Instrument readout model
    """
    squid_nei = Variable()
    read_noise_frac = Variable(default=0.1)
    dwell_time = Variable()
    revisit_rate = Variable()
    nyquist_inductance = Variable()
