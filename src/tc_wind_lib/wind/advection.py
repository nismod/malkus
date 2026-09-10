import numba
import numpy as np


@numba.njit
def lin_chavas_2012(
    eye_heading_deg: float,
    eye_speed_ms: float,
    hemisphere: int,
    alpha: float = 0.56,
    beta: float = 19.2,
) -> np.complex128:
    """
    Calculate the advective wind vector

    For reconstruction of realistic advective wind component, (and rationale
    for alpha and beta) see section 2 of: Lin, N., and D. Chavas (2012), On
    hurricane parametric wind and applications in storm surge modeling, J.
    Geophys. Res., 117, D09120, doi:10.1029/2011JD017126

    Arguments:
        eye_heading_deg: Heading of eye in degrees clockwise from north
        eye_speed_ms: Speed of eye in metres per second
        hemisphere: +1 for northern, -1 for southern
        alpha: Fractional reduction of advective wind speed from eye speed
        beta: Degrees advective winds tend to rotate past storm track (in direction of rotation)

    Returns:
        np.complex128: Advective wind vector
    """

    # Bearing of advective component (storm track heading with beta correction)
    phi_a: float = np.radians(eye_heading_deg - hemisphere * beta)

    # Absolute magnitude of vector is eye speed decreased by alpha factor
    mag_v_a: float = eye_speed_ms * alpha

    # Find components
    return mag_v_a * np.sin(phi_a) + mag_v_a * np.cos(phi_a) * 1j