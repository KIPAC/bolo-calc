""" Computations for noise estimation """

import numpy as np
import pickle as pk
import os
import io

from . import physics

def Flink(n, Tb, Tc):
    """
    Link factor for the bolo to the bath

    Args:
    n (float): thermal carrier index
    Tb (float): bath temperature [K]
    Tc (float): transition temperature [K]
    """
    return (((n + 1)/(2 * n + 3)) * (1 - (Tb / Tc)**(2 * n + 3)) /
            (1 - (Tb / Tc)**(n + 1)))

def G(psat, n, Tb, Tc):
    """
    Thermal conduction between the bolo and the bath

    Args:
    psat (float): saturation power [W]
    n (float): thermal carrier index
    Tb (float): bath temperature [K]
    Tc (float): bolo transition temperature [K]
    """
    return (psat * (n + 1) * (Tc**n) /
            ((Tc**(n + 1)) - (Tb**(n + 1))))


def calc_photon_NEP(popts, freqs, factors=None):
    """
    Calculate photon NEP [W/rtHz] for a detector

    Args:
    popts (list): power from elements in the optical elements [W]
    freqs (list): frequencies of observation [Hz]
    """
    popt = np.sum(popts, axis=0)
    #popt = sum([x for x in popts])
    # Don't consider correlations
    if factors is None:
        popt2 = popt*popt
        nep = np.sqrt(np.trapz((2. * physics.h * freqs * popt + 2. * popt2), freqs))
        neparr = nep
        return nep, neparr

    popt2 = sum([popts[i] * popts[j]
                 for i in range(len(popts))
                     for j in range(len(popts))])
    popt2arr = sum([factors[i] * factors[j] * popts[i] * popts[j]
                        for i in range(len(popts))
                        for j in range(len(popts))])
    nep = np.sqrt(np.trapz((2. * physics.h * freqs * popt + 2. * popt2), freqs))
    neparr = np.sqrt(np.trapz((2. * physics.h * freqs * popt + 2. * popt2arr), freqs))
    return nep, neparr

def bolo_NEP(flink, G_val, Tc):
    """
    Thremal carrier NEP [W/rtHz]

    Args:
    flink (float): link factor to the bolo bath
    G (float): thermal conduction between the bolo and the bath [W/K]
    Tc (float): bolo transition temperature [K]
    """
    return np.sqrt(4 * physics.kB * flink * (Tc**2) * G_val)

def read_NEP(pelec, boloR, nei, sfact=1.):
    """
    Readout NEP [W/rtHz] for a voltage-biased bolo

    Args:
    pelec (float): bias power [W]
    boloR (float): bolometer resistance [Ohms]
    nei (float): noise equivalent current [A/rtHz]
    """
    responsivity = sfact / np.sqrt(boloR * pelec)
    return nei / responsivity

def dPdT(eff, freqs):
    """
    Change in power on the detector with change in CMB temperature [W/K]

    Args:
    eff (float): detector efficiency
    freqs (float): observation frequencies [Hz]
    """
    temp = np.array([physics.Tcmb for f in freqs])
    return np.trapz(physics.ani_pow_spec(
        np.array(freqs), temp, np.array(eff)), freqs)

def NET_from_NEP(nep, freqs, sky_eff, opt_coup=1.0):
    """
    NET [K-rts] from NEP

    Args:
    nep (float): NEP [W/rtHz]
    freqs (list): observation frequencies [Hz]
    sky_eff (float): efficiency between the detector and the sky
    opt_coup (float): optical coupling to the detector. Default to 1.
    """
    dpdt = opt_coup * dPdT(sky_eff, freqs)
    return nep / (np.sqrt(2.) * dpdt)

def NET_arr(net, n_det, det_yield=1.0):
    """
    Array NET [K-rts] from NET per detector and num of detectors

    Args:
    net (float): NET per detector
    n_det (int): number of detectors
    det_yield (float): detector yield. Defaults to 1.
    """
    return net/(np.sqrt(n_det * det_yield))

def map_depth(net_arr, fsky, tobs, obs_eff):
    """
    Sensitivity [K-arcmin] given array NET

    Arg:
    net_arr (float): array NET [K-rts]
    fsky (float): sky fraction
    tobs (float): observation time [s]
    """
    return np.sqrt(
        (4. * physics.PI * fsky * 2. * np.power(net_arr, 2.)) /
        (tobs * obs_eff)) * (10800. / physics.PI)


class Noise: #pylint: disable=too-many-instance-attributes
    """
    Noise object calculates NEP, NET, mapping speed, and sensitivity

    Args:
    phys (src.Physics): parent Physics object

    Parents:
    phys (src.Physics): Physics object
    """
    def __init__(self):
        """ Constructor """

        # Aperture stop names
        self._ap_names = ["APERT", "STOP", "LYOT"]

        # Correlation files
        corr_dir = os.path.join(
            os.path.split(__file__)[0], "PKL")
        self._p_c_apert, self._c_apert = pk.load(io.open(
            os.path.join(corr_dir, "coherentApertCorr.pkl"), "rb"),
                            encoding="latin1")
        self._p_c_stop,  self._c_stop = pk.load(io.open(
            os.path.join(corr_dir, "coherentStopCorr.pkl"), "rb"),
                            encoding="latin1")
        self._p_i_apert, self._i_apert = pk.load(io.open(
            os.path.join(corr_dir, "incoherentApertCorr.pkl"), "rb"),
                            encoding="latin1")
        self._p_i_stop,  self._i_stop = pk.load(io.open(
            os.path.join(corr_dir, "incoherentStopCorr.pkl"), "rb"),
                            encoding="latin1")

        # Detector pitch array
        self._det_p = self._p_c_apert
        # Geometric pitch factor
        self._geo_fact = 6  # Hex packing

    def corr_facts(self, elems, det_pitch, ap_names, flamb_max=3.):
        """
        Calculate the Bose white-noise correlation factor

        Args:
        elems (list): optical elements in the camera
        det_pitch (float): detector pitch in f-lambda units
        flamb_max (float): the maximum detector pitch distance
        for which to calculate the correlation factor.
        Default is 3.
        """
        ndets = int(round(flamb_max / (det_pitch), 0))
        #import pdb
        #pdb.set_trace()
        inds1 = [np.argmin(abs(np.array(self._det_p) -
                 det_pitch * (n + 1)))
                 for n in range(ndets)]
        inds2 = [np.argmin(abs(np.array(self._det_p) -
                 det_pitch * (n + 1) * np.sqrt(3.)))
                 for n in range(ndets)]
        inds = np.sort(inds1 + inds2)
        at_det = False
        factors = []
        for elem_ in elems:
            if at_det:
                factors.append(1.)
                continue
            if "CMB" in elem_:
                use_abs = abs(self._c_apert)
            elif elem_ in ap_names:
                use_abs = abs(self._i_stop)
                at_det = True
            else:
                use_abs = abs(self._i_apert)
            factors.append(np.sqrt(1. + self._geo_fact*(np.sum([use_abs[ind] for ind in inds]))))

        return np.array(factors)

    def photon_NEP(self, popts, freqs, **kwargs):
        """
        Calculate photon NEP [W/rtHz] for a detector

        Args:
        popts (list): power from elements in the optical elements [W]
        freqs (list): frequencies of observation [Hz]
        elems (list): optical elements
        det_pitch (float): detector pitch in f-lambda units. Default is None.
        """
        #popt = sum([x for x in popts])
        # Don't consider correlations
        elems = kwargs.get('elems', None)
        det_pitch = kwargs.get('det_pitch', None)
        ap_names = kwargs.get('ap_names', None)
        if elems is None or det_pitch is None:
            factors = None
        else:
            factors = self.corr_facts(elems, det_pitch, ap_names)
        return calc_photon_NEP(popts, freqs, factors)
