"""Parametric wind profiles."""

from typing import Protocol

import numba
import numpy as np


class WindProfile(Protocol):
    """A radial wind profile evaluated around a cyclone eye."""

    def __call__(
        self,
        r_m: np.ndarray,
        *,
        v_max_ms: float,
        r_max_m: float,
        min_pressure_pa: float,
        env_pressure_pa: float,
        lat_deg: float,
    ) -> np.ndarray: ...


@numba.njit(error_model="numpy")
def holland_1980(
    r_m: np.ndarray,
    *,
    v_max_ms: float,
    r_max_m: float,
    min_pressure_pa: float,
    env_pressure_pa: float,
    lat_deg: float,
) -> np.ndarray:
    """Return Holland (1980) wind speeds at radii in metres."""

    molar_mass_air = 0.02897
    gas_constant = 8.314
    temperature_k = 293.0
    density = min_pressure_pa * molar_mass_air / (gas_constant * temperature_k)
    omega = 2 * np.pi / (24 * 60 * 60)
    coriolis = np.abs(2 * omega * np.sin(np.radians(lat_deg)))
    pressure_deficit = env_pressure_pa - min_pressure_pa
    beta = (
        v_max_ms**2 * np.e * density
        + coriolis * v_max_ms * r_max_m * np.e * density
    ) / pressure_deficit
    winds = np.sqrt(
        (
            (r_max_m / r_m) ** beta
            * beta
            * pressure_deficit
            * np.exp(-(r_max_m / r_m) ** beta)
            + r_m**2 * coriolis**2 / 4
        )
        / density
    ) - coriolis * r_m / 2
    return np.clip(winds, 0, None)
