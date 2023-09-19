class Constants:

    _G: float = 6.67430e-11

    _MASS_UNITS = {"kg": 1,
                   "sm": 1.988e30}

    _DISP_UNITS = {"m": 1,
                   "au": 1.496e+11}

    @classmethod
    @property
    def G(cls) -> float:
        return cls._G

    @classmethod
    @property
    def MASS_UNITS(cls) -> float:
        return cls._MASS_UNITS

    @classmethod
    @property
    def DISP_UNITS(cls) -> float:
        return cls._DISP_UNITS
