""" Instrument readout model """

from cfgmdl import Model, Parameter

class Readout(Model):
    """
    Instrument readout model
    """
    squid_nei = Parameter()
    read_noise_frac = Parameter(default=0.1)
    dwell_time = Parameter()
    revisit_rate = Parameter()
    nyquist_inductance = Parameter()
