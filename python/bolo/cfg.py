""" Class derived from cfgmdl used to configure bolo-calc"""

import numpy as np
import scipy.stats as sps

from collections import OrderedDict as odict

from copy import deepcopy

from cfgmdl import Property, Parameter, ParamHolder
from cfgmdl.utils import is_none, defaults_decorator

from .pdf import ChoiceDist
from .interp import FreqIntrep
from .utils import cfg_path

class ParamOrInterpHolder(ParamHolder):
    """ Allows parameter values to be interpolated using a frequency repsonse file"""

    interp = Property(dtype=str, default=None, help="Interoplation file")

    def __init__(self, *args, **kwargs):
        """ Constuctor """
        super(ParamOrInterpHolder, self).__init__(*args, **kwargs)
        self._sampled_values = None
        self._cached_interps = None

    @staticmethod
    def _channel_value(arr, chan_idx):
        """ Picks the value for a paricular channel,

        This allows using a single parameter to represent multiple channels
        """
        if chan_idx is None:
            return arr
        if np.isscalar(arr):
            return arr
        if isinstance(arr, FreqIntrep):
            return arr
        if arr.size < 2:
            return arr
        return arr[chan_idx]

    def _cache_interps(self):
        """ Cache the interpolator object """
        self._cached_interps = None
        if self.interp is None:
            return
        tokens = self.interp.split(',')
        if len(tokens) == 1:
            self._cached_interps = FreqIntrep(cfg_path(tokens[0]))
        else:
            self._cached_interps = np.array([FreqIntrep(cfg_path(token)) for token in tokens])

    def rvs(self, freqs, nsamples, chan_idx=0):
        """ Sample values

        This just returns the sampled values.
        It does not store them.
        """
        val = self._channel_value(self.value, chan_idx)
        if self.interp is None or not nsamples:
            return val
        errs = self._channel_value(self.errors, chan_idx)
        self._cache_interps()
        if self._cached_interps is None:
            if np.isnan(errs):
                return val
            return sps.norm(loc=val, scale=errs).rvs(nsamples).reshape((nsamples, 1))
        interp = self._channel_value(self._cached_interps, chan_idx)
        return interp.rvs(freqs, nsamples).reshape((nsamples, len(freqs)))

    def sample(self, freqs, nsamples, chan_idx=None):
        """ Sample values

        This stores the sampled values.
        """
        self._sampled_values = self.rvs(freqs, nsamples, chan_idx)
        return self._sampled_values

    def unsample(self):
        """ This removes the stored sampled values"""
        self._sampled_values = None

    def __call__(self):
        """Return the product of the value and the scale

        This uses the stored sampled values if they are present.
        """
        if self._sampled_values is not None:
            return self._sampled_values*self.scale
        return super(ParamOrInterpHolder, self).__call__()


class ParamOrInterp(Parameter):
    """ Property sub-class to allow parameter values
    to be interpolated using a frequency repsonse file
    """

    defaults = deepcopy(Parameter.defaults)
    defaults['dtype'] = (ParamOrInterpHolder, "Data type")

    @defaults_decorator(defaults)
    def __init__(self, **kwargs):
        super(ParamOrInterp, self).__init__(**kwargs)


class ParamOrPDFHolder(ParamHolder):
    """ Allows parameter values to be sampled using a PDF or means and errors """

    pdf = Property(dtype=str, default=None, help="PDF file")

    def __init__(self, *args, **kwargs):
        """ Constuctor """
        super(ParamOrPDFHolder, self).__init__(self, *args, **kwargs)
        self._sampled_values = None

    def sampler(self):
        """ Construct and return an object to do the sampling"""
        if self.pdf is None:
            if np.isnan(self.errors).any():
                return None
            return sps.norm(loc=self.value, scale=self.errors)
        return ChoiceDist(cfg_path(self.pdf))

    def rvs(self, nsamples):
        """ Sample values

        This just returns the sampled values.
        It does not store them.
        """
        if nsamples or self.pdf is not None:
            pdf = self.sampler()
        else:
            pdf = None
        if pdf is None:
            return None
        if nsamples:
            return pdf.rvs(nsamples)
        return pdf.mean()

    def sample(self, nsamples):
        """ Sample values

        This stores the sampled values.
        """
        self._sampled_values = self.rvs(nsamples)
        return self._sampled_values

    def unsample(self):
        """ This removes the stored sampled values"""
        self._sampled_values = None

    def __call__(self):
        """Return the product of the value and the scale

        This uses the stored sampled values if they are present.
        """
        if self._sampled_values is not None:
            return self._sampled_values*self.scale
        return super(ParamOrPDFHolder, self).__call__()


class ParamOrPdf(Parameter):
    """ Allows parameter values to be sampled using a PDF or means and errors """

    defaults = deepcopy(Parameter.defaults)
    defaults['dtype'] = (ParamOrPDFHolder, "Data type")

    @defaults_decorator(defaults)
    def __init__(self, **kwargs):
        super(ParamOrPdf, self).__init__(**kwargs)

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

    def __str__(self):
        """ A pretty represenation of the stats """
        return "%.3g +- [%.3g %.3g] %s" % (self._mean, self._deltas[1], self._deltas[2], self._unit_name)

    def todict(self):
        """ Put the summary stats into a dictionary """
        o_dict = odict()
        for vn in ['_mean', '_median', '_std']:
            o_dict["%s%s" % (self._name, vn)] = [self.__dict__[vn]]
        for idx, vn in enumerate(['_n_2_sig', '_n_1_sig', '_p_1_sig', '_p_2_sig']):
            o_dict["%s%s" % (self._name, vn)] = [self._quantiles[idx]]
        return o_dict



class Output(Property):
    """ A property sub-class that will convert values back
    from SI units to input units.

    Also has a function to provide summary statistics
    """

    defaults = deepcopy(Property.defaults)
    defaults['outname'] = (None, "Output Name")

    def __get__(self, obj, objtype=None):
        """Get the value from the client object

        Parameter
        ---------
        obj : ...
            The client object

        Return
        ------
        out : ...
            The requested value
        """
        attr = getattr(obj, self.private_name)
        if self.unit is None or is_none(attr):
            return attr
        return self.unit.inverse(attr) #pylint: disable=not-callable

    def summarize(self, obj):
        """ Compute and return the summary statistics """
        unit_name = ""
        val = getattr(obj, self.private_name)
        if self.unit:
            unit_name = self.unit.name
            val = self.unit.inverse(val)
        return StatsSummary(self.public_name, val, unit_name)
