



class Unit:
    """
    Object for handling unit conversions

    Args:
    unit (str): name of the unit

    Attributes:
    name (str): where 'unit' arg is stored
    """

    to_SI_dict = {
        "GHz": 1.e+09,
        "mm": 1.e-03,
        "aW/rtHz": 1.e-18,
        "pA/rtHz": 1.e-12,
        "pW": 1.e-12,
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
        "K^2": 1.,
        "s": 1.,
        "deg": 1,
        "NA": 1.
        }
        # Check that passed unit is available

    def __init__(self, unit):
        # Dictionary of SI unit conversions
        # Check that passed unit is available
        if unit is None:
            self._SI = 1.
        if isinstance(unit, str):
            if unit not in self.to_SI_dict:
                raise KeyError("Passed unit '%s' not understood by \
                    Unit object" % (unit))
            else:
                self._SI = self.to_SI_dict[unit]
        elif isinstance(unit, float):
            self._SI = unit

    def to_SI(self, val):
        """Convert value to SI unit """
        return val * self._SI

    def from_SI(self, val):
        """ Convert value from SI unit """
        return val / self._SI

    @property
    def SI(self):
        return self.value * self._SI
