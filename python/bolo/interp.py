"""
Interpolates quantities vs frequency
"""


import numpy as np

from cfgmdl.utils import is_none

from .utils import read_txt_to_np


class FreqIntrep:
    """
    Interpolates quantities vs frequency, with errrors
    for detectors and optics
    """
    def __init__(self, fname):
        """ Constructor """
        self.fname = fname
        band_data = read_txt_to_np(self.fname)
        if len(band_data) == 3:
            self.freq, self.tran, self.errs = band_data
        elif len(band_data) == 2:
            self.freq, self.tran = band_data
            self.errs = None
        self.freq *= 1e9
        self.tran_interp = None
        self.errs_interp = None


    def mean(self):
        """ Return the weighted mean of the interpolation curve """
        return np.mean(self.freq*self.tran)


    def cache_grid(self, freqs):
        """ Cache the values and errors from the interpolation grid """
        if freqs is None:
            self.tran_interp = self.tran
            self.errs_interp = self.errs
            return
        mask = np.bitwise_and(freqs < self.freq[-1], freqs > self.freq[0])
        self.tran_interp = np.where(mask, np.interp(freqs, self.freq, self.tran), 0.).clip(0., 1.)
        if is_none(self.errs):
            self.errs_interp = None
            return
        self.errs_interp = np.where(mask, np.interp(freqs, self.freq, self.errs), 0.).clip(1e-6, np.inf)
        return


    def rvs(self, freqs, nsample=0):
        """ Sample values """

        self.cache_grid(freqs)
        if not nsample:
            return self.tran_interp

        if is_none(self.errs_interp):
            return (np.ones((nsample, 1)) * self.tran_interp).clip(0., 1.)
        return np.random.normal(self.tran_interp, self.errs_interp, (nsample, len(self.tran_interp))).clip(0., 1.)
