""" Sky model """

from collections import OrderedDict as odict

import numpy as np

import h5py as hp

from cfgmdl import Model, Property, Parameter, cached

from .utils import is_not_none, cfg_path
from . import physics
#from .unit import Unit
from .cfg import Variable

GHz_to_Hz = 1.e+09
m_to_mm = 1.e+03
mm_to_um = 1.e+03


def interp_spectra(freqs, freq_grid, vals):
    """ Interpolate a spectrum """
    freq_grid = freq_grid * GHz_to_Hz
    return np.interp(freqs, freq_grid, vals)


class AtmModel:
    """ Atmospheric model using tabulated values"""

    def __init__(self, fname, site):
        """ Constructor """
        self._file = hp.File(fname, 'r')
        self._data = self._file[site]

    @staticmethod
    def get_keys(pwv, elev):
        """ Get the keys in the tabulated data """
        return ["%d,%d" % (int(round(pwv_ * m_to_mm, 1) * mm_to_um), int(round(elev_, 0))) for pwv_, elev_ in np.broadcast(pwv, elev)]

    def temp(self, keys, freqs):
        """ Get interpolated temperatures """
        return np.array([interp_spectra(freqs, self._data[key_][0], self._data[key_][2]) for key_ in keys])

    def trans(self, keys, freqs):
        """ Get interpolated transmission coefs """
        return np.array([interp_spectra(freqs, self._data[key_][0], self._data[key_][3]) for key_ in keys])


class CustomAtm:
    """ Atmospheric model using custom value from a txt file """

    def __init__(self, fname):
        """ Constructor """
        self._freqs, self._temps, self._trans = np.loadtxt(fname, unpack=True, usecols=[0, 2, 3], dtype=np.float)

    def temp(self, freqs):
        """ Get interpolated temperatures """
        return interp_spectra(freqs, self._freqs, self._temps)

    def trans(self, freqs):
        """ Get interpolated transmission coefs """
        return interp_spectra(freqs, self._freqs, self._trans)


class Atmosphere(Model):
    """ Atmosphere model """
    atm_model_file = Property(dtype=str)

    def __init__(self, **kwargs):
        """ Constructor """
        self._atm_model = None
        self._telescope = None
        self._sampled_keys = None
        self._nsamples = 1
        super(Atmosphere, self).__init__(**kwargs)

    def set_telescope(self, value):
        """ Set the telescope

        This is needed to sample elevation and PWV values
        """
        self._telescope = value

    @cached(uses=[atm_model_file])
    def cached_model(self):
        """ Cache the Atmosphere model """
        if is_not_none(self.atm_model_file):
            return AtmModel(cfg_path(self.atm_model_file), self._telescope.site)
        if is_not_none(self._telescope.atm_file):
            return CustomAtm(cfg_path(self._telescope.atm_file))
        return None

    def sample(self, nsamples):
        """ Sample the atmosphere """
        model = self.cached_model
        if isinstance(model, CustomAtm):
            self._sampled_keys = None
            self._nsamples = 1
            return
        self._telescope.pwv.sample(nsamples)
        self._telescope.elevation.sample(nsamples)
        self._sampled_keys = model.get_keys(1e-6*np.atleast_1d(self._telescope.pwv()), np.atleast_1d(self._telescope.elevation())) #pylint: disable=no-member
        self._nsamples = max(nsamples, 1)

    def temp(self, freqs):
        """ Get sampled temperatures """
        model = self.cached_model
        nfreqs = len(freqs)
        if self._sampled_keys is None:
            out_shape = (1, 1, nfreqs)
            return model.temp(freqs).reshape(out_shape)  #pylint: disable=no-member
        out_shape = (self._nsamples, 1, nfreqs)
        return model.temp(self._sampled_keys, freqs).reshape(out_shape)  #pylint: disable=no-member

    def trans(self, freqs):
        """ Get sampled transmission coefs """
        model = self.cached_model
        nfreqs = len(freqs)
        if self._sampled_keys is None:
            out_shape = (1, 1, nfreqs)
            return model.trans(freqs).reshape(out_shape)  #pylint: disable=no-member
        out_shape = (self._nsamples, 1, nfreqs)
        return model.trans(self._sampled_keys, freqs).reshape(out_shape)  #pylint: disable=no-member


class Foreground(Model):
    """
    Foreground model base class
    """
    spectral_index = Parameter(required=True, help="Powerlaw index")
    scale_frequency = Parameter(required=True, help="Frequency", unit="GHz")
    emiss = Parameter(default=1., help="Emissivity")


class Dust(Foreground):
    """
    Dust emission model
    """
    amplitude = Variable(required=True, help="Dust amplitude", unit="MJy")
    scale_temperature = Variable(required=True, help="Temperature", unit='K')

    def __init__(self, **kwargs):
        """ Constructor """
        self._nsamples = 1
        self.amp = None
        self.scale_temp = None
        super(Dust, self).__init__(**kwargs)

    def sample(self, nsamples):
        """ Sample this component """
        self.amplitude.sample(nsamples)
        self.scale_temperature.sample(nsamples)

        self.amp = np.expand_dims(np.expand_dims(self.amplitude.SI, -1), -1)
        self.scale_temp = np.expand_dims(np.expand_dims(self.scale_temperature.SI, -1), -1)
        self._nsamples = max(self.amp.size, self.scale_temp.size)

    def temp(self, freqs):
        """ Get sampled temperatures """
        out_shape = (self._nsamples, 1, len(freqs))
        return self.__temp(freqs, self.emiss.SI, self.amp, self.scale_frequency.SI, self.spectral_index.SI, self.scale_temp).reshape(out_shape)
        #self.__temp(freqs).reshape(out_shape)

    #@Function
    @staticmethod
    def __temp(freqs, emiss, amp, scale_frequency, spectral_index, scale_temp): #pylint: disable=too-many-arguments
        """
        Return the galactic effective physical temperature
        """
        # Passed amplitude [W/(m^2 sr Hz)] converted from [MJy]
        amp = emiss * amp
        # Frequency scaling
        # (freq / scale_freq)**dust_ind
        if np.isfinite(scale_frequency).all() and np.isfinite(spectral_index).all():
            freq_scale = (freqs / scale_frequency)**(spectral_index)
        else:
            freq_scale = 1.
        # Effective blackbody scaling
        # BB(freq, dust_temp) / BB(dust_freq, dust_temp)
        if np.isfinite(scale_temp).all() and np.isfinite(scale_frequency).all():
            spec_scale = physics.bb_spec_rad(freqs, scale_temp) / physics.bb_spec_rad(scale_frequency, scale_temp)
        else:
            spec_scale = 1.
        # Convert [W/(m^2 sr Hz)] to brightness temperature [K_RJ]
        pow_spec_rad = amp * freq_scale * spec_scale
        return physics.Tb_from_spec_rad(freqs, pow_spec_rad)


class Synchrotron(Foreground):
    """
    Synchrotron emission model
    """
    amplitude = Variable(required=True, help="Dust amplitude", unit="K_RJ")

    def __init__(self, **kwargs):
        """ Constructor """
        self.amp = None
        self._nsamples = 1
        super(Synchrotron, self).__init__(**kwargs)

    def sample(self, nsamples):
        """ Sample this component """
        self.amplitude.sample(nsamples)
        self.amp = np.expand_dims(np.expand_dims(self.amplitude.SI, -1), -1)
        self._nsamples = self.amp.size

    def temp(self, freqs):
        """ Get sampled temperatures """
        out_shape = (self._nsamples, 1, len(freqs))
        return self.__temp(freqs, self.emiss.SI, self.amp, self.scale_frequency.SI, self.spectral_index.SI).reshape(out_shape)
        #self.__temp(freqs).reshape(out_shape)

    @staticmethod
    def __temp(freqs, emiss, amp, scale_frequency, spectral_index):
        """
        Return the effective physical temperature
        """
        bright_temp = emiss * amp
        # Frequency scaling (freq / sync_freq)**sync_ind
        freq_scale = (freqs / scale_frequency)**spectral_index
        scaled_bright_temp = bright_temp * freq_scale
        # Convert brightness temperature [K_RJ] to physical temperature [K]
        return physics.Tb_from_Trj(freqs, scaled_bright_temp)


class Universe(Model):
    """
    Collection of emission models
    """
    dust = Property(dtype=Dust, help='Dust model')
    synchrotron = Property(dtype=Synchrotron, help='Synchrotron model')
    atmosphere = Property(dtype=Atmosphere, help='Atmospheric model')

    sources = ['cmb', 'dust', 'synchrotron', 'atmosphere']

    def sample(self, nsamples):
        """ Sample the sky component """
        self.dust.sample(nsamples)
        self.synchrotron.sample(nsamples)
        self.atmosphere.sample(nsamples)

    def temp(self, freqs):
        """ Get sampled temperatures """
        ret = odict()
        ret['cmb'] = physics.Tcmb
        ret['dust'] = self.dust.temp(freqs)
        ret['synchrotron'] = self.synchrotron.temp(freqs)
        ret['atmosphere'] = self.atmosphere.temp(freqs)
        return ret

    def trans(self, freqs):
        """ Get sampled transmission coefs """
        ret = odict()
        ret['atmosphere'] = self.atmosphere.trans(freqs)
        return ret
