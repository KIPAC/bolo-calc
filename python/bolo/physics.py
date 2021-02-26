"""
Physics object calculates physical quantities

Attributes:
h (float): Planck constant [J/s]
kB (float): Boltzmann constant [J/K]
c (float): Speed of light [m/s]
PI (float): Pi
mu0 (float): Permability of free space [H/m]
ep0 (float): Permittivity of free space [F/m]
Z0 (float): Impedance of free space [Ohm]
Tcmb (float): CMB Temperature [K]
co (dict): CO Emission lines [Hz]
"""

import numpy as np
#import jax.numpy as jnp
jnp = np

h = 6.6261e-34
kB = 1.3806e-23
c = 299792458.0
PI = 3.14159265
mu0 = 1.256637e-6
ep0 = 8.854188e-12
Z0 = np.sqrt(mu0/ep0)
Tcmb = 2.725

_ = """
def check_inputs(x, inputs=None):
    ret = []
    if isinstance(x, np.ndarray) or isinstance(x, list):
        x = np.array(x).astype(np.float)
        ones = np.ones(x.shape)
        if inputs is None:
            return x
        ret.append(x)
        for inp in inputs:
            if callable(inp):
                ret.append(inp(x).astype(np.float))
            elif isinstance(inp, np.ndarray) or isinstance(inp, list):
                ret.append(np.array(inp).astype(np.float)*ones)
            elif isinstance(inp, int) or isinstance(inp, float):
                ret.append(np.array([inp for i in x]).astype(np.float)*ones)
            else:
                raise Exception(
                    "Non-numeric value %s passed in Physics" % (str(x)))
    elif isinstance(x, int) or isinstance(x, float):
        if inputs is None:
            return x
        ret.append(float(x))
        for inp in inputs:
            if callable(inp):
                ret.append(float(inp(x)))
            elif isinstance(inp, int) or isinstance(inp, float):
                ret.append(float(inp))
            else:
                ret.append(inp)
    else:
        raise Exception(
            "Non-numeric value %s passed in Physics" % (str(x)))
    return ret
"""


def lamb(freq, ind=1.0):
    """
    Convert from from frequency [Hz] to wavelength [m]

    Args:
    freq (float): frequencies [Hz]
    ind: index of refraction. Defaults to 1
    """
    #freq, ind = _check_inputs(freq, [ind])
    return c/(freq*ind)

def band_edges(freqs, tran):
    """ Find the -3 dB points of an arbirary band """
    max_tran = jnp.amax(tran)
    max_tran_loc = jnp.argmax(tran)
    lo_point = jnp.argmin(
        abs(tran[:max_tran_loc] - 0.5 * max_tran))
    hi_point = jnp.argmin(
        abs(tran[max_tran_loc:] - 0.5 * max_tran)) + max_tran_loc
    flo = freqs[lo_point]
    fhi = freqs[hi_point]
    return flo, fhi

def spill_eff(freq, pixd, fnum, wf=3.0):
    """
    Pixel beam coupling efficiency given a frequency [Hz],
    pixel diameter [m], f-number, and beam wasit factor

    Args:
    freq (float): frequencies [Hz]
    pixd (float): pixel size [m]
    fnum (float): f-number
    wf (float): waist factor. Defaults to 3.
    """
    #freq, pixd, fnum, wf = _check_inputs(freq, [pixd, fnum, wf])
    return 1. - jnp.exp(
        (-jnp.power(np.pi, 2)/2.) * jnp.power(
            (pixd / (wf * fnum * (c/freq))), 2))

def edge_taper(ap_eff):
    """
    Edge taper given an aperture efficiency

    Args:
    ap_eff (float): aperture efficiency
    """
    return 10. * jnp.log10(1. - ap_eff)

def apert_illum(freq, pixd, fnum, wf=3.0):
    """
    Aperture illumination efficiency given a frequency [Hz],
    pixel diameter [m], f-number, and beam waist factor

    Args:
    freq (float): frequencies [Hz]
    pixd (float): pixel diameter [m]
    fnum (float): f-number
    wf (float): beam waist factor
    """
    #freq, pixd, fnum, wf = _check_inputs(freq, [pixd, fnum, wf])
    lamb_val = lamb(freq)
    w0 = pixd / wf
    theta_stop = lamb_val / (np.pi * w0)
    theta_apert = jnp.arange(0., jnp.arctan(1. / (2. * fnum)), 0.01)
    V = jnp.exp(-jnp.power(theta_apert, 2.) / jnp.power(theta_stop, 2.))
    eff_num = jnp.power(
        jnp.trapz(V * jnp.tan(theta_apert / 2.), theta_apert), 2.)
    eff_denom = jnp.trapz(
        jnp.power(V, 2.) * jnp.sin(theta_apert), theta_apert)
    eff_fact = 2. * jnp.power(jnp.tan(theta_apert/2.), -2.)
    return (eff_num / eff_denom) * eff_fact

def ruze_eff(freq, sigma):
    """
    Ruze efficiency given a frequency [Hz] and surface RMS roughness [m]

    Args:
    freq (float): frequencies [Hz]
    sigma (float): RMS surface roughness
    """
    #freq, sigma = _check_inputs(freq, [sigma])
    return jnp.exp(-jnp.power(4 * np.pi * sigma / (c / freq), 2.))

def ohmic_eff(freq, sigma):
    """
    Ohmic efficiency given a frequency [Hz] and conductivity [S/m]

    Args:
    freq (float): frequencies [Hz]
    sigma (float): conductivity [S/m]
    """
    #freq, sigma = _check_inputs(freq, [sigma])
    return 1. - 4. * jnp.sqrt(np.pi * freq * mu0 / sigma) / Z0

#def brightness_temp(freq, spec_rad):
#    """
#    Brightness temperature [K_RJ] given a frequency [Hz] and
#    spectral radiance [W Hz^-1 sr^-1 m^-2]
#
#    Args:
#    freq (float): frequency [Hz]
#    spec_rad (float): spectral radiance [W Hz^-1 sr^-1 m^-2]
#    """
#    #return spec_rad / (2 * kB * (freq / c)**2)
#    return spec_rad / (kB * (freq / c)**2)

def Trj_over_Tb(freq, Tb):
    """
    Brightness temperature [K_RJ] given a physical temperature [K]
    and frequency [Hz]. dTrj / dTb

    Args:
    freq (float): frequencies [Hz]
    Tb (float): physical temperature. Default to Tcmb
    """
    #freq, Tb = _check_inputs(freq, [Tb])
    x = (h * freq)/(Tb * kB)
    thermo_fact = jnp.power(
        (jnp.exp(x) - 1.), 2.) / (jnp.power(x, 2.) * jnp.exp(x))
    return 1. / thermo_fact

def Tb_from_spec_rad(freq, pow_spec):
    """
    FIXME
    """
    return (
        (h * freq / kB) /
        jnp.log((2 * h * (freq**3 / c**2) / pow_spec) + 1))

def Tb_from_Trj(freq, Trj):
    """
    FIXME
    """
    alpha = (h * freq) / kB
    return alpha / jnp.log((2 * alpha / Trj) + 1)

def inv_var(err):
    """
    Inverse variance weights based in input errors

    Args:
    err (float): errors to generate weights
    """
    jnp.seterr(divide='ignore')
    return 1. / (jnp.sqrt(jnp.sum(1. / (jnp.power(np.array(err), 2.)))))

def dielectric_loss(freq, thick, ind, ltan):
    """
    The dielectric loss of a substrate given the frequency [Hz],
    substrate thickness [m], index of refraction, and loss tangent

    Args:
    freq (float): frequencies [Hz]
    thick (float): substrate thickness [m]
    ind (float): index of refraction
    ltan (float): loss tangent
    """
    #freq, thick, ind, ltan = _check_inputs(freq, [thick, ind, ltan])
    return 1. - jnp.exp(
        (-2. * PI * ind * ltan * thick) / (lamb(freq)))

def rj_temp(powr, bw, eff=1.0):
    """
    RJ temperature [K_RJ] given power [W], bandwidth [Hz], and efficiency

    Args:
    powr (float): power [W]
    bw (float): bandwidth [Hz]
    eff (float): efficiency
    """
    return powr / (kB * bw * eff)

def n_occ(freq, temp):
    """
    Photon occupation number given a frequency [Hz] and
    blackbody temperature [K]

    freq (float): frequency [Hz]
    temp (float): blackbody temperature [K]
    """
    #freq, temp = _check_inputs(freq, [temp])
    fact = (h * freq)/(kB * temp)
    fact = jnp.where(fact > 100, 100, fact)
    #with jnp.errstate(divide='raise'):
    return 1. / (jnp.exp(fact) - 1.)

def a_omega(freq):
    """
    Throughput [m^2] for a diffraction-limited detector
    given the frequency [Hz]

    Args:
    freq (float): frequencies [Hz]
    """
    #freq = _check_inputs(freq)
    return lamb(freq)**2

def bb_spec_rad(freq, temp, emis=1.0):
    """
    Blackbody spectral radiance [W/(m^2 sr Hz)] given a frequency [Hz],
    blackbody temperature [K], and blackbody emissivity

    Args:
    freq (float): frequencies [Hz]
    temp (float): blackbody temperature [K]
    emiss (float): blackbody emissivity. Defaults to 1.
    """
    #freq, temp, emis = _check_inputs(freq, [temp, emis])
    return (emis * (2 * h * (freq**3) /
            (c**2)) * n_occ(freq, temp))

def bb_pow_spec(freq, temp, emis=1.0):
    """
    Blackbody power spectrum [W/Hz] on a diffraction-limited polarimeter
    for a frequency [Hz], blackbody temperature [K],
    and blackbody emissivity

    Args:
    freq (float): frequencies [Hz]
    temp (float): blackbody temperature [K]
    emiss (float): blackbody emissivity. Defaults to 1.
    """
    #freq, temp, emis = _check_inputs(freq, [temp, emis])
    return 0.5 * a_omega(freq) * bb_spec_rad(freq, temp, emis)

def ani_pow_spec(freq, temp, emiss=1.0):
    """
    Derivative of blackbody power spectrum with respect to blackbody
    temperature, dP/dT, on a diffraction-limited detector [W/K] given
    a frequency [Hz], blackbody temperature [K], and blackbody
    emissivity

    Args:
    freq (float): frequency [Hz]
    temp (float): blackbody temperature [K]
    emiss (float): blackbody emissivity, Defaults to 1.
    """
    #freq, temp, emiss = _check_inputs(freq, [temp, emiss])
    return emiss * kB * jnp.exp((h * freq)/(kB * temp)) * (h*freq*n_occ(freq, temp) / kB*temp)**2

def pow_frac(T1, T2, freqs):
    """ Fractional power between two physical temperatures """
    return bb_pow_spec(freqs, T1) / bb_pow_spec(freqs, T2)
