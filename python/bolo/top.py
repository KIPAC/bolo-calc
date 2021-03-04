"""Top level configuration"""


from cfgmdl import Property, Model
from .sky import Universe
from .instrument import Instrument
from .utils import set_config_dir

class SimConfig(Model):
    """Simulation configuration"""
    nsky_sim = Property(dtype=int, default=0)
    ndet_sim = Property(dtype=int, default=0)
    save_summary = Property(dtype=bool, default=True)
    save_sim = Property(dtype=bool, default=True)
    save_optical = Property(dtype=bool, default=True)
    freq_resol = Property(dtype=float, default=None)
    config_dir = Property(dtype=str, default="../config")

    def __init__(self, **kwargs):
        """ Constructor """
        super(SimConfig, self).__init__(**kwargs)
        set_config_dir(self.config_dir)


class Top(Model):
    """Top level configuration"""
    sim_config = Property(dtype=SimConfig)
    universe = Property(dtype=Universe)
    instrument = Property(dtype=Instrument)

    def __init__(self, **kwargs):
        """ Constructor """
        self.universe = None
        self.instrument = None
        super(Top, self).__init__(**kwargs)
        self.universe.atmosphere.set_telescope(self.instrument)

    def run(self):
        """ Run the entire analysis """
        self.instrument.run(self.universe, self.sim_config)
