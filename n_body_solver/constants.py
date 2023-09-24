class Constants:

    _G: float = 6.67430e-11

    _MASS_UNITS = {"kg": 1,
                   "sm": 1.988e30}

    _DISP_UNITS = {"m": 1,
                   "au": 1.496e+11}

    _VELOCITY_UNITS = {"mps": 1,
                       "kmps": 1e3}

    _TIME_UNITS = {"s": 1,
                   "year": 3.154e+7}

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

    @classmethod
    @property
    def VELOCITY_UNITS(cls) -> float:
        return cls._VELOCITY_UNITS

    @classmethod
    @property
    def TIME_UNITS(cls) -> float:
        return cls._TIME_UNITS
