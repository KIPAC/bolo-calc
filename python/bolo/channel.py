""" Channel model """

import numpy as np

from cfgmdl import Property, Parameter, Model

from .utils import is_not_none, is_none
#from . import physics, noise
from . import noise
from .sky import Universe
#from .unit import Unit
from .cfg import Variable

class Channel(Model):  #pylint: disable=too-many-instance-attributes
    """ Channel Model """

    _min_tc_tb_diff = 0.010

    band_center = Parameter(unit="GHz")
    fractional_bandwidth = Parameter(default=.35)
    band_response = Variable(default=1.)

    det_eff = Variable(default=1.)
    squid_nei = Variable(default=1., unit='pA/rtHz')
    bolo_resistance = Variable(default=1., unit='Ohm')

    pixel_size = Parameter(default=6.8, unit='mm')
    waist_factor = Parameter(default=3.)

    Tc = Parameter(default=.165, unit='K')
    Tc_fraction = Parameter()

    num_det_per_water = Property(dtype=int, default=542)
    num_wafer_per_optics_tube = Property(dtype=int, default=1)
    num_optics_tube = Property(dtype=int, default=3)

    psat = Parameter()
    psat_factor = Parameter(default=3.)

    read_frac = Parameter(default=0.1)
    carrier_index = Parameter(default=3)
    G = Parameter(unit='pW/K')
    Flink = Parameter()
    Yield = Parameter()
    response_factor = Parameter()
    nyquist_inductance = Parameter()

    def __init__(self, **kwargs):
        """ Constructor """
        self._optical_effic = None
        self._optical_emiss = None
        self._optical_temps = None
        self._sky_temp_dict = None
        self._sky_tran_dict = None
        self._det_effic = None
        self._det_emiss = None
        self._det_temp = None
        self._camera = None
        self._idx = None
        super(Channel, self).__init__(**kwargs)
        self._freqs = None
        self.bandwidth = None

    def set_camera(self, camera, idx):
        """ Set the parent camera and the channel index """
        self._camera = camera
        self._idx = idx

    def sample(self, nsamples):
        """ Sample PDF parameters """
        self.det_eff.sample(nsamples)
        self.squid_nei.sample(nsamples)
        self.bolo_resistance.sample(nsamples)

    @property
    def camera(self):
        """ Return the parent camera """
        return self._camera

    @property
    def freqs(self):
        """ Return the evaluation frequencies """
        return self._freqs

    @property
    def ndet(self):
        """ Return the total number of detectors per channel """
        return self.num_det_per_water*self.num_wafer_per_optics_tube*self.num_optics_tube

    @property
    def idx(self):
        """ Return the channel index """
        return self._idx

    def photon_NEP(self, elem_power):
        """ Return the photon NEP given the power in the element in the optical chain """
        #wave_number = self.pixel_size / self.camera.f_number * physics.lamb(self.band_center)
        return noise.calc_photon_NEP(elem_power, self._freqs)

    def bolo_NEP(self, opt_pow):
        """ Return the bolometric NEP given the detector details """
        tb = self._camera.bath_temperature()
        tc = self.Tc()
        n = 1. #self.n
        if is_not_none(self.G) and np.isfinite(self.G()).all():
            g = self.G()
        else:
            if is_not_none(self.psat_factor) and np.isfinite(self.psat_factor()):
                p_sat = opt_pow * self.psat_factor()
            else:
                p_sat = self.psat()
            g = noise.G(p_sat, n, tb, tc)
        if is_not_none(self.Flink) and np.isfinite(self.Flink()):
            flink = self.Flink()
        else:
            flink = noise.Flink(n, tb, tc)
        return noise.bolo_NEP(flink, g, tc)

    def read_NEP(self, opt_pow):
        """ Return the readout NEP given the detector details """
        if np.isnan(self.squid_nei()).any():
            return None
        if np.isnan(self.bolo_resistance()).any():
            return None
        if is_not_none(self.psat) and np.isfinite(self.psat()).all():
            p_stat = self.psat()
        else:
            p_stat = self.psat_factor() * opt_pow
        p_bias = (p_stat - opt_pow).clip(0, np.inf)
        if is_not_none(self.response_factor) and np.isfinite(self.response_factor()).all():
            s_fact = self.response_factor()
        else:
            s_fact = 1.        
        return noise.read_NEP(p_bias, self.bolo_resistance().T, self.squid_nei().T, s_fact)

    def compute_evaluation_freqs(self, freq_resol=None):
        """ Compute and return the evaluation frequencies """
        self.bandwidth = self.band_center() * self.fractional_bandwidth()
        if freq_resol is None:
            freq_resol = 0.25*self.bandwidth
        else:
            freq_resol = freq_resol * self.band_center()
        fmin = self.band_center() - 0.5*self.bandwidth
        fmax = self.band_center() + 0.5*self.bandwidth
        freq_step = np.ceil(self.bandwidth/freq_resol).astype(int)
        return np.linspace(fmin, fmax, freq_step+1)

    def eval_optical_chain(self, nsample=0, freq_resol=None):
        """ Evaluate the performance of the optical chain for this channel """
        self._freqs = self.compute_evaluation_freqs(freq_resol)
        self._optical_effic = []
        self._optical_emiss = []
        self._optical_temps = []
        for elem in self._camera.optics.values():
            effic, emiss, temps = elem.compute_channel(self, self._freqs, nsample)
            self._optical_effic.append(effic)
            self._optical_emiss.append(emiss)
            self._optical_temps.append(temps)

    def eval_det_response(self, nsample=0, freq_resol=None):
        """ Evaluate the detector response for this channel """
        self._freqs = self.compute_evaluation_freqs(freq_resol)
        self.band_response.sample(nsample, self._freqs)
        self.det_eff.sample(nsample)
        #def_eff_shaped = np.expand_dims(self.det_eff(), -1)
        self._det_effic = self.band_response() * self.det_eff()
        self._det_emiss = 0.
        self._det_temp = self._camera.bath_temperature()

    def eval_sky(self, universe, freq_resol=None):
        """ Evaluate the sky parameters for this channel

        This is done here, b/c the frequencies we care about are chanel dependent.
        """
        self._freqs = self.compute_evaluation_freqs(freq_resol)
        self._sky_temp_dict = universe.temp(self._freqs)
        self._sky_tran_dict = universe.trans(self._freqs)

    @property
    def optical_effic(self):
        """ Return the optical element efficiecies for this channel """
        return self._optical_effic

    @property
    def optical_emiss(self):
        """ Return the optical element emissivities for this channel """
        return self._optical_emiss

    @property
    def optical_temps(self):
        """ Return the optical element temperatures for this channel """
        return self._optical_temps

    @property
    def sky_temps(self):
        """ Return the sky component temperatures for this channel """
        return [ self._sky_temp_dict.get(k) for k in Universe.sources ]

    @property
    def sky_effic(self):
        """ Return the sky component efficiecies for this channel """
        return [ self._sky_tran_dict.get(k, 1.) for k in Universe.sources ]

    @property
    def sky_emiss(self):
        """ Return the sky component emissivities for this channel """
        return [1] * len(Universe.sources)
