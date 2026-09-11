"""Source-model wind conventions for tropical-cyclone track catalogues."""

from enum import StrEnum


class WindSpeedReference(StrEnum):
    """Reference frame of input maximum wind speeds."""

    EARTH_RELATIVE = "earth_relative"
    EYE_RELATIVE = "eye_relative"


class TrackSource(StrEnum):
    """Supported source model families with known wind conventions."""

    IBTRACS = "ibtracs"
    IRIS = "iris"
    STORM = "storm"
    CHAZ = "chaz"
    EMANUEL = "emanuel"


WIND_SPEED_REFERENCE_BY_SOURCE = {
    TrackSource.IBTRACS: WindSpeedReference.EARTH_RELATIVE,
    TrackSource.IRIS: WindSpeedReference.EARTH_RELATIVE,
    TrackSource.STORM: WindSpeedReference.EARTH_RELATIVE,
    TrackSource.CHAZ: WindSpeedReference.EYE_RELATIVE,
    TrackSource.EMANUEL: WindSpeedReference.EYE_RELATIVE,
}


IS_SYNTHETIC_BY_SOURCE = {
    TrackSource.IBTRACS: False,
    TrackSource.IRIS: True,
    TrackSource.STORM: True,
    TrackSource.CHAZ: True,
    TrackSource.EMANUEL: True,
}
