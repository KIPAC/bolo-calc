
from collections import OrderedDict as odict

import numpy as np

#from bolo.ctrl.param import Property, Derived, Parameter, Model
from bolo.model.dummy import Property, Model

from bolo.ctrl.utils import is_not_none
from bolo.calc import physics

class Foreground(Model):
    """
    Foreground object contains the foreground parameters for the sky
    """    
    spectral_index = Property(dtype=float, required=True, format="%.2e", help="Powerlaw index")
    amplitude = Property(dtype=float, required=True, format="%.2e", help="Foreground amplitude")
    scale_frequency = Property(dtype=float, required=True, format="%.2e", help="Frequency")

        
class Dust(Foreground):

    scale_temperature = Property(dtype=float, format="%.2e", help="Frequency")

    def temp(self, freq, emiss=1.0): 
        """
        Return the galactic effective physical temperature

        Args:
        freq (float): frequency at which to evaluate the physical temperature
        emiss (float): emissivity of the galactic dust. Default to 1.
        """
        # Passed amplitude [W/(m^2 sr Hz)] converted from [MJy]
        freq = np.array(freq)
        emiss = np.array(emiss)
        amp = emiss * self.amplitude
        # Frequency scaling
        # (freq / scale_freq)**dust_ind
        if is_not_none(self.scale_frequency) and is_not_none(self.spectral_index):
            freq_scale = (freq / self.scale_frequency)**(self.spectral_index)
        else:
            freq_scale = 1.
        # Effective blackbody scaling
        # BB(freq, dust_temp) / BB(dust_freq, dust_temp)
        if is_not_none(self.scale_temperature) and is_not_none(self.scale_fequency):
            spec_scale = physics.bb_spec_rad(freq, self.scale_temperature) / physics.bb_spec_rad(self.scale_frequency.val, self.scale_temperature.val)
        else:
            spec_scale = 1.
        # Convert [W/(m^2 sr Hz)] to brightness temperature [K_RJ]
        pow_spec_rad = amp * freq_scale * spec_scale
        # Convert brightness temperature [K_RJ] to physical temperature [K]
        phys_temp = physics.Tb_from_spec_rad(freq, pow_spec_rad)
        return phys_temp



class Synchrotron(Foreground):

    def temp(self, freq, emiss=1.0):
        """
        Return the synchrotron spectral radiance [W/(m^2-Hz)]

        Args:
        freq (float): frequency at which to evaluate the spectral radiance
        emiss (float): emissivity of the synchrotron radiation. Default to 1.
        """
        # Passed brightness temp [K_RJ]
        freq = np.array(freq)
        emiss = np.array(emiss)
        bright_temp = emiss * self.amplitude
        # Frequency scaling (freq / sync_freq)**sync_ind
        freq_scale = (freq / self.scale_frequency)**self.spectral_index
        scaled_bright_temp = bright_temp * freq_scale
        # Convert brightness temperature [K_RJ] to physical temperature [K]
        phys_temp = physics.Tb_from_Trj(freq, scaled_bright_temp)
        return phys_temp



class Universe(Model):

    dust = Property(dtype=Dust, help='Dust model')
    synchrotron = Property(dtype=Synchrotron, help='Synchrotron model')
    

