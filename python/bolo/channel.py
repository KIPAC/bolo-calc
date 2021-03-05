""" Channel model """

import numpy as np

from cfgmdl import Property, Model

from .utils import is_not_none
from . import physics, noise
from .sky import Universe
#from .unit import Unit
from .cfg import Variable

class Channel(Model):  #pylint: disable=too-many-instance-attributes
    """ Channel Model """

    _min_tc_tb_diff = 0.010

    band_center = Variable(unit="GHz")
    fractional_bandwidth = Variable(default=.35)
    band_response = Variable(default=1.)

    det_eff = Variable(default=1.)
    squid_nei = Variable(default=1., unit='pA/rtHz')
    bolo_resistance = Variable(default=1., unit='Ohm')

    pixel_size = Variable(default=6.8, unit='mm')
    waist_factor = Variable(default=3.)

    Tc = Variable(default=.165, unit='K')
    Tc_fraction = Variable()

    num_det_per_water = Property(dtype=int, default=542)
    num_wafer_per_optics_tube = Property(dtype=int, default=1)
    num_optics_tube = Property(dtype=int, default=3)

    psat = Variable()
    psat_factor = Variable(default=3.)

    read_frac = Variable(default=0.1)
    carrier_index = Variable(default=3)
    G = Variable(unit='pW/K')
    Flink = Variable()
    Yield = Variable()
    response_factor = Variable()
    nyquist_inductance = Variable()

    noise_calc = noise.Noise()

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
        self._flo = None
        self._fhi = None
        self._freq_mask = None
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
    def flo(self):
        """ Return the -3dB point"""
        return self._flo

    @property
    def fhi(self):
        """ Return the +3dB point """
        return self._fhi

    @property
    def ndet(self):
        """ Return the total number of detectors per channel """
        return self.num_det_per_water*self.num_wafer_per_optics_tube*self.num_optics_tube

    @property
    def idx(self):
        """ Return the channel index """
        return self._idx

    def photon_NEP(self, elem_power, elems=None, ap_names=None):
        """ Return the photon NEP given the power in the element in the optical chain """
        if elems is None:
            return self.noise_calc.photon_NEP(elem_power, self._freqs)
        det_pitch = self.pixel_size.SI / (self.camera.f_number.SI * physics.lamb(self.band_center.SI))
        return self.noise_calc.photon_NEP(elem_power, self._freqs, elems=elems, det_pitch=det_pitch, ap_names=ap_names)

    def bolo_NEP(self, opt_pow):
        """ Return the bolometric NEP given the detector details """
        tb = self._camera.bath_temperature()
        tc = self.Tc.SI
        n = 1. #self.n
        if is_not_none(self.G) and np.isfinite(self.G.SI).all():
            g = self.G.SI
        else:
            if is_not_none(self.psat_factor) and np.isfinite(self.psat_factor.SI):
                p_sat = opt_pow * self.psat_factor.SI
            else:
                p_sat = self.psat.SI
            g = noise.G(p_sat, n, tb, tc)
        if is_not_none(self.Flink) and np.isfinite(self.Flink.SI):
            flink = self.Flink.SI
        else:
            flink = noise.Flink(n, tb, tc)
        return noise.bolo_NEP(flink, g, tc)

    def read_NEP(self, opt_pow):
        """ Return the readout NEP given the detector details """
        if np.isnan(self.squid_nei.SI).any():
            return None
        if np.isnan(self.bolo_resistance.SI).any():
            return None
        if is_not_none(self.psat) and np.isfinite(self.psat.SI).all():
            p_stat = self.psat.SI
        else:
            p_stat = self.psat_factor.SI * opt_pow
        p_bias = (p_stat - opt_pow).clip(0, np.inf)
        if is_not_none(self.response_factor) and np.isfinite(self.response_factor.SI).all():
            s_fact = self.response_factor.SI
        else:
            s_fact = 1.
        return noise.read_NEP(p_bias, self.bolo_resistance.SI.T, self.squid_nei.SI.T, s_fact)

    def compute_evaluation_freqs(self, freq_resol=None):
        """ Compute and return the evaluation frequencies """
        self.bandwidth = self.band_center.SI * self.fractional_bandwidth.SI
        if freq_resol is None:
            freq_resol = 0.05*self.bandwidth
        else:
            freq_resol = freq_resol * 1e9
        self._flo = self.band_center.SI - 0.5*self.bandwidth
        self._fhi = self.band_center.SI + 0.5*self.bandwidth
        freq_step = np.ceil(self.bandwidth/freq_resol).astype(int)

        self._freqs = np.linspace(self._flo, self._fhi, freq_step+1)
        band_mean_response = self.band_response.sample(0, self._freqs)
        if np.isscalar(band_mean_response):
            return self._freqs
        self._flo, self._fhi = physics.band_edges(self._freqs, band_mean_response)
        self.bandwidth = self._fhi - self._flo
        freq_mask = np.bitwise_and(self._freqs >= self._flo, self._freqs <= self._fhi)
        self._freqs = self._freqs[freq_mask]
        return self._freqs

    def eval_optical_chain(self, nsample=0, freq_resol=None):
        """ Evaluate the performance of the optical chain for this channel """
        self.compute_evaluation_freqs(freq_resol)
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
        self._det_effic = self.band_response.SI * self.det_eff.SI
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
    def sky_names(self):
        """ Return the list of the names of the sky components """
        return list(self._sky_temp_dict.keys())

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
