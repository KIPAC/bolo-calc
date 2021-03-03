""" Class derived from cfgmdl used to configure bolo-calc"""

import numpy as np
import scipy.stats as sps

from collections import OrderedDict as odict
from collections.abc import Iterable

from copy import deepcopy

from cfgmdl import Property, Parameter, ParamHolder, Choice
from cfgmdl.utils import is_none, defaults_decorator

from .pdf import ChoiceDist
from .interp import FreqIntrep
from .utils import cfg_path

class VariableHolder(ParamHolder):
    """ Allows parameter values to be interpolated using a frequency repsonse file"""

    fname = Property(dtype=str, default=None, help="Data file")
    var_type = Choice(choices=['pdf', 'dist', 'gauss', 'const'], default='const')

    def __init__(self, *args, **kwargs):
        """ Constuctor """
        self._value = None
        super(VariableHolder, self).__init__(*args, **kwargs)
        self._sampled_values = None
        self._cached_interps = None

    @staticmethod
    def _channel_value(arr, chan_idx):
        """ Picks the value for a paricular channel,

        This allows using a single parameter to represent multiple channels
        """
        if not isinstance(arr, Iterable):
            return arr
        if not arr.shape:
            return arr
        if len(arr) < 2:
            return arr[0]
        return arr[chan_idx]

    def _cache_interps(self, freqs=None):
        """ Cache the interpolator object """
        if self.var_type == 'const':
            self._cached_interps = None
            return
        if self.var_type == 'gauss':
            if np.isnan(self.errors).any():
                self._cached_interps = None
                return

            self._cached_interps = np.array([ sps.norm(loc=val_, scale=sca_) for val_, sca_ in zip(np.atleast_1d(self.value), np.atleast_1d(self.errors))])
            return
        tokens = self.fname.split(',')
        if self.var_type == 'pdf':
            self._cached_interps = np.array([ChoiceDist(cfg_path(token)) for token in tokens])
            self._value = np.array([ pdf.mean() for pdf in self._cached_interps ])
            return
        self._cached_interps = np.array([FreqIntrep(cfg_path(token)) for token in tokens])
        if freqs is None:
            self._value = np.array([ pdf.mean_trans() for pdf in self._cached_interps ])
            return
        self._value = np.array([ pdf.rvs(freqs) for pdf in self._cached_interps ])


    def rvs(self, nsamples, freqs=None, chan_idx=0):
        """ Sample values

        This just returns the sampled values.
        It does not store them.
        """
        self._cache_interps(freqs)
        val = self._channel_value(self.value, chan_idx)
        if self._cached_interps is None or not nsamples:
            return val
        interp = self._channel_value(self._cached_interps, chan_idx)
        if self.var_type == 'gauss':
            return interp.rvs(nsamples).reshape((nsamples, 1))
        if self.var_type == 'pdf':
            return interp.rvs(nsamples).reshape((nsamples, 1))
        return interp.rvs(freqs, nsamples).reshape((nsamples, len(freqs)))

    def sample(self, nsamples, freqs=None, chan_idx=0):
        """ Sample values

        This stores the sampled values.
        """
        self._sampled_values = self.rvs(nsamples, freqs, chan_idx)
        return self.SI

    def unsample(self):
        """ This removes the stored sampled values"""
        self._sampled_values = None

    @property
    def scaled(self):
        """Return the product of the value and the scale

        This uses the stored sampled values if they are present.
        """
        if self._sampled_values is not None:
            return self._sampled_values*self.scale
        return super(VariableHolder, self).scaled


class Variable(Parameter):
    """ Property sub-class to allow parameter values
    to be interpolated using a frequency repsonse file
    """

    defaults = deepcopy(Parameter.defaults)
    defaults['dtype'] = (VariableHolder, "Data type")

    @defaults_decorator(defaults)
    def __init__(self, **kwargs):
        super(Variable, self).__init__(**kwargs)

    def _cast_type(self, value, obj=None):
        """Hook took override type casting"""
        if is_none(value):
            return None
        return value


class StatsSummary:
    """ Summarize the statistical properties of the distribution of computed
    parameter"""

    def __init__(self, name, vals, unit_name="", axis=None):
        """ Constructor """
        self._name = name
        self._unit_name = unit_name
        self._mean = np.mean(vals, axis=axis)
        self._median = np.median(vals, axis=axis)
        self._std = np.std(vals, axis=axis)
        self._quantiles = np.quantile(vals, [0.023, 0.159, 0.841, 0.977], axis=axis)
        self._deltas = np.abs(self._quantiles - self._median)

    def element_string(self, idx):
        """ A pretty represenation of the stats for one element """
        return "%0.4f +- [%0.4f %0.4f] %s" % (self._mean[idx], self._deltas[1,idx], self._deltas[2,idx], self._unit_name)

    def __str__(self):
        """ A pretty represenation of the stats """
        return "%0.4f +- [%0.4f %0.4f] %s" % (self._mean, self._deltas[1], self._deltas[2], self._unit_name)

    def todict(self):
        """ Put the summary stats into a dictionary """
        o_dict = odict()
        for vn in ['_mean', '_median', '_std']:
            o_dict["%s%s" % (self._name, vn)] = [self.__dict__[vn]]
        for idx, vn in enumerate(['_n_2_sig', '_n_1_sig', '_p_1_sig', '_p_2_sig']):
            o_dict["%s%s" % (self._name, vn)] = [self._quantiles[idx]]
        return o_dict



class Output(Parameter):
    """ A property sub-class that will convert values back
    from SI units to input units.

    Also has a function to provide summary statistics
    """

    outname = Property(dtype=str, default=None)

    def summarize(self, obj):
        """ Compute and return the summary statistics """
        unit_name = ""
        val = getattr(obj, self.private_name).value
        return StatsSummary(self.public_name, val, unit_name)


    def summarize_by_element(self, obj):
        """ Compute and return the summary statistics """
        unit_name = ""
        val = getattr(obj, self.private_name).value
        val = val.reshape((val.shape[0], np.product(val.shape[1:])))
        return StatsSummary(self.public_name, val, unit_name, axis=1)
