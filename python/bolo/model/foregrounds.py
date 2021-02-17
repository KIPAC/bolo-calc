
from collections import OrderedDict as odict

import numpy as np
from jax import grad, jacfwd, jacrev
import jax.numpy as jnp

from cfgmdl import Property, Model, Parameter, Function

from bolo.ctrl.utils import is_not_none
from bolo.calc import physics

class Foreground(Model):
    """
    Foreground model base class
    """    
    spectral_index = Property(dtype=float, required=True, format="%.2e", help="Powerlaw index")
    amplitude = Parameter(required=True, help="Foreground amplitude", bounds=[0., 3.])
    scale_frequency = Parameter(required=True, help="Frequency")

        
class Dust(Foreground):
    """
    Dust emission model
    """

    scale_temperature = Property(dtype=float, format="%.2e", help="Frequency")
    
    @Function
    def temp(freq, emiss, amplitude, scale_frequency, spectral_index, scale_temperature): 
        """
        Return the galactic effective physical temperature

        Args:
        freq (float): frequency at which to evaluate the physical temperature
        emiss (float): emissivity of the galactic dust. Default to 1.
        """
        # Passed amplitude [W/(m^2 sr Hz)] converted from [MJy]
        amplitude = emiss * amplitude
        # Frequency scaling
        # (freq / scale_freq)**dust_ind
        if np.isfinite(scale_frequency) and np.isfinite(spectral_index):
            freq_scale = (freq / scale_frequency)**(spectral_index)
        else:
            freq_scale = 1.
        # Effective blackbody scaling
        # BB(freq, dust_temp) / BB(dust_freq, dust_temp)
        if np.isfinite(scale_temperature) and np.isfinite(scale_frequency):
            spec_scale = physics.bb_spec_rad(freq, scale_temperature) / physics.bb_spec_rad(scale_frequency, scale_temperature)
        else:
            spec_scale = 1.
        # Convert [W/(m^2 sr Hz)] to brightness temperature [K_RJ]
        pow_spec_rad = amplitude * freq_scale * spec_scale
        return physics.Tb_from_spec_rad(freq, pow_spec_rad)

        
    
class Synchrotron(Foreground):
    """
    Synchrotron emission model
    """

    @Function
    def temp(freq, emiss, amplitude, scale_frequency, spectral_index):
        bright_temp = emiss * amplitude
        # Frequency scaling (freq / sync_freq)**sync_ind
        freq_scale = (freq / scale_frequency)**spectral_index
        scaled_bright_temp = bright_temp * freq_scale
        # Convert brightness temperature [K_RJ] to physical temperature [K]
        return physics.Tb_from_Trj(freq, scaled_bright_temp)
    

class Universe(Model):
    """
    Collection of emission models
    """
    dust = Property(dtype=Dust, help='Dust model')
    synchrotron = Property(dtype=Synchrotron, help='Synchrotron model')
    

