""" Model of optical elements """

from collections import OrderedDict as odict

import numpy as np

from cfgmdl import Parameter, Model
from cfgmdl.tools import build_class
from cfgmdl.utils import is_not_none

from .cfg import Variable
from . import physics


class ChannelResults:
    """ Performance parameters for one optical element for one channel """

    def __init__(self):
        """ Constructor """
        self.temp = None
        self.refl = None
        self.spil = None
        self.scat = None
        self.spil_temp = None
        self.scat_temp = None
        self.abso = None
        self.emiss = None
        self.effic = None

    #@Function
    @staticmethod
    def emission(freqs, abso, spil, spil_temp, scat, scat_temp, temp): #pylint: disable=too-many-arguments
        """ Compute the emission for this element """
        return abso +\
          spil * physics.pow_frac(spil_temp, temp, freqs) +\
          scat * physics.pow_frac(scat_temp, temp, freqs)

    #@Function
    @staticmethod
    def efficiency(refl, abso, spil, scat):
        """ Compute the transmission for this element """
        return (1-refl)*(1-abso)*(1-spil)*(1-scat)

    def calculate(self, freqs):
        """ Compute the results for the frequencies of interest for a given channel """
        emiss_shape = np.broadcast(freqs, self.abso, self.spil, self.spil_temp, self.scat, self.scat_temp, self.temp).shape
        self.emiss = self.emission(freqs, self.abso, self.spil, self.spil_temp, self.scat, self.scat_temp, self.temp).reshape(emiss_shape)
        effic_shape = np.broadcast(self.refl, self.abso, self.spil, self.scat).shape
        self.effic = self.efficiency(self.refl, self.abso, self.spil, self.scat).reshape(effic_shape)

    def __call__(self):
        """ Return key parameters """
        return (self.effic, self.emiss, self.temp)


class OpticalElement(Model):
    """Model for a single optical element"""

    temperature = Variable(required=True)
    spillover_temp = Parameter(unit="K")
    scatter_temp = Parameter(unit="K")
    surface_rough = Parameter()

    absorption = Variable(required=True)
    reflection = Variable(required=True)
    spillover = Variable(required=True)
    scatter_frac = Variable(required=True)

    def __init__(self, **kwargs):
        """ Constructor """
        super(OpticalElement, self).__init__(**kwargs)
        self.elem_name = None
        self.results = dict()

    def unsample(self):
        """ Clear out the samples parameters """
        self.temperature.unsample()
        self.reflection.unsample()
        self.spillover.unsample()
        self.scatter_frac.unsample()

    def sample(self, freqs, nsample, chan_idx):
        """ Sample input parameters for a given channel """
        self.temperature.sample(nsample)
        results_ = ChannelResults()
        results_.temp = self.temperature.SI
        results_.refl = self.reflection.sample(nsample, freqs, chan_idx)
        results_.spil = self.spillover.sample(nsample, freqs, chan_idx)
        if is_not_none(self.surface_rough) and np.isfinite(self.surface_rough()).all():
            results_.scat = 1. - physics.ruze_eff(freqs, self.surface_rough)
        else:
            results_.scat = self.scatter_frac.sample(nsample, freqs, chan_idx)
        if is_not_none(self.spillover_temp) and np.isfinite(self.spillover_temp()).all():
            results_.spil_temp = self.spillover_temp.SI
        else:
            results_.spil_temp = results_.temp
        if is_not_none(self.scatter_temp) and np.isfinite(self.scatter_temp()).all():
            results_.scat_temp = self.scatter_temp.SI
        else:
            results_.scat_temp = results_.temp
        self.results[chan_idx] = results_
        return results_

    def compute_channel(self, channel, freqs, nsample):
        """ Compute the results for the frequencies of interest for a given channel """
        self.unsample()
        results_ = self.sample(freqs, nsample, channel.idx)
        results_.abso = self.calc_abso(channel, freqs, nsample)
        results_.calculate(freqs)
        return results_()

    def calc_abso(self, channel, freqs, nsample):
        """ Compute the absorption for a given channel """
        return self.absorption.sample(nsample, freqs, channel.idx)


class Mirror(OpticalElement):
    """ OpticalElement sub-class for mirrors """

    conductivity = Parameter()

    def calc_abso(self, channel, freqs, nsample):
        """ Compute the absorption for a given channel """
        if is_not_none(self.conductivity) and np.isfinite(self.conductivity()).all():
            return 1. - physics.ohmic_eff(freqs, self.conductivity())
        return super(Mirror, self).calc_abso(channel, freqs, nsample)


class Dielectric(OpticalElement):
    """ OpticalElement sub-class for dielectrics """

    thickness = Parameter()
    index = Parameter()
    loss_tangent = Parameter()

    def calc_abso(self, channel, freqs, nsample):
        """ Compute the absorption for a given channel """
        if is_not_none(self.thickness) and is_not_none(self.index) and is_not_none(self.loss_tangent):
            return physics.dielectric_loss(freqs, self.thickness(), self.index(), self.loss_tangent())
        return super(Dielectric, self).calc_abso(channel, freqs, nsample)


class ApertureStop(OpticalElement):
    """ OpticalElement sub-class for apertures """

    def calc_abso(self, channel, freqs, nsample):
        """ Compute the absorption for a given channel """
        pixel_size = channel.pixel_size()
        f_number = channel.camera.f_number()
        waist_factor = channel.waist_factor()

        if is_not_none(pixel_size) and is_not_none(f_number) and is_not_none(waist_factor):
            return 1. - physics.spill_eff(np.array(freqs), pixel_size, f_number, waist_factor)
        return super(ApertureStop, self).calc_abso(channel, freqs, nsample)


class Optics_Base(Model):
    """Base class for optical chains"""

    def __init__(self, **kwargs):
        """ Constructor """

        super(Optics_Base, self).__init__(**kwargs)
        self.elements = odict()
        self.mirrors = odict()
        self.dielectics = odict()
        self.apertureStops = odict()
        for key, val in self.__dict__.items():
            if isinstance(val, OpticalElement):
                val.elem_name = key
                self.elements[key] = val
            if isinstance(val, Mirror):
                self.mirrors[key] = val
            if isinstance(val, Dielectric):
                self.dielectics[key] = val
            if isinstance(val, ApertureStop):
                self.apertureStops[key] = val



def build_optics_class(name="Optics", **kwargs):
    """ Build a class that consists of a set of OpticalElements

    Parameter
    ---------
    name : `str`
        The name of the new class
    kwargs : Heirachical dictionary used to build elements


    Returns
    -------
    optics_class : `type`
        The new class, which has all the requested properties

    """
    type_dict = {None:OpticalElement, 'Mirror':Mirror, 'Dielectric':Dielectric, 'ApertureStop':ApertureStop}
    return build_class(name, (Optics_Base, ), [kwargs], [type_dict])
