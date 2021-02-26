""" Instrument readout model """

from cfgmdl import Property, Model, Parameter

class Readout(Model):
    """
    Instrument readout model
    """
    squid_nei = Property(dtype=float)
    read_noise_frac = Parameter(default=0.1)
    dwell_time = Property(dtype=float)
    revisit_rate = Property(dtype=float)
    nyquist_inductance = Property(dtype=float)
