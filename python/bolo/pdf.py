""" Simple implementation of PDFs for instrument parameters"""
import numpy as np

from .utils import read_txt_to_np

class ChoiceDist:
    """
    ChoiceDist object holds probability distribution functions (PDFs)
    for instrument parameters

    Parameters
    ----------
    inp (str or arr): file name for the input PDF or input data array

    """
    def __init__(self, inp):
        """
        Parameters
        ----------
        inp (str or arr): file name for the input PDF or input data array
        """
        if isinstance(inp, str):
            self._fname = inp
            self._inp = read_txt_to_np(inp)
        else:
            self._fname = None
            self._inp = np.array(inp)

        if len(self._inp.shape) != 2:
            raise ValueError("ChoiceDist requires 2 input arrays %s" % self._inp.shape)


        self.val = self._inp[0]
        self.prob = self._inp[1]
        self.prob /= np.sum(self.prob)
        self._cum = np.cumsum(self.prob)

    # ***** Public Methods *****
    def rvs(self, nsample=1):
        """
        Samle the distribution nsample times

        Args:
        nsample (int): the number of times to sample the distribution
        """
        if nsample == 1:
            return np.random.choice(self.val, size=nsample, p=self.prob)[0]
        return np.random.choice(self.val, size=nsample, p=self.prob)

    def change(self, new_avg):
        """ Arithmetically shift the distribution to the new central value """
        old_mean = self.mean()
        shift = new_avg - old_mean
        self.val += shift

    def mean(self):
        """ Return the mean of the distribution """
        if self.prob is not None:
            return np.sum(self.prob * self.val)
        return np.mean(self.val)

    def std(self):
        """ Return the standard deviation of the distribution """
        if self.prob is not None:
            mean = self.mean()
            return np.sqrt(np.sum(self.prob * ((self.val - mean) ** 2)))
        return np.std(self.val)

    def median(self):
        """ Return the median of the distribution """
        if self.prob is not None:
            arg = np.argmin(abs(self._cum - 0.5))
            return self.val[arg]
        return np.median(self.val)

    def one_sigma(self):
        """ Return the 15.9% and 84.1% values """
        med = self.median()
        if self.prob is not None:
            lo, hi = np.interp((0.159, 0.841), self._cum, self.val)
        else:
            lo, hi = np.percentile(self.val, [0.159, 0.841])
        return (hi-med, med-lo)

    def two_sigma(self):
        """ Return the 2.3% and 97.7% values """
        med = self.median()
        if self.prob is not None:
            lo, hi = np.interp((0.023, 0.977), self._cum, self.val)
        else:
            lo, hi = np.percentile(self.val, [0.023, 0.977])
        return (hi-med, med-lo)
