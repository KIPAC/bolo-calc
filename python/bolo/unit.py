""" Unit conversions """

from cfgmdl import Unit

to_SI_dict = {
    "GHz": 1.e+09,
    "mm": 1.e-03,
    "aW/rtHz": 1.e-18,
    "pA/rtHz": 1.e-12,
    "pW": 1.e-12,
    "pW/K": 1.e-12,
    "um": 1.e-06,
    "pct": 1.e-02,
    "uK": 1.e-06,
    "uK-rts": 1.e-06,
    "uK-amin": 1.e-06,
    "uK^2": 1.e-12,
    "yr": (365.25*24.*60.*60),
    "e-4": 1.e-04,
    "e+6": 1.e+06,
    "um RMS": 1.e-06,
    "MJy": 1.e-20,
    "Ohm": 1.,
    "W/Hz": 1.,
    "Hz": 1.,
    "m": 1.,
    "W/rtHz": 1.,
    "A/rtHz": 1.,
    "W": 1.,
    "K": 1.,
    "K_RJ": 1.,
    "K^2": 1.,
    "s": 1.,
    "deg": 1,
    "NA": 1.
    }

Unit.update(to_SI_dict)
