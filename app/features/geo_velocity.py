"""
Haversine geographic distance and impossible travel velocity calculation.
"""

import math
from typing import Tuple
from app.config import EARTH_RADIUS_KM, IMPOSSIBLE_TRAVEL_SPEED_KMH, MIN_TIME_DELTA_HOURS

RAD_FACTOR: float = math.pi / 180.0
HALF_RAD_FACTOR: float = math.pi / 360.0


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius_km: float = EARTH_RADIUS_KM,
) -> float:
    """
    Calculate the great-circle distance between two geographic coordinates in kilometers.

    Handles international date line meridian wrap-around for longitudes (wrapping delta
    into the [-180.0, 180.0] degree range). Latitudes are clamped to [-90.0, 90.0].

    Parameters
    ----------
    lat1 : float
        Latitude of first point in degrees.
    lon1 : float
        Longitude of first point in degrees.
    lat2 : float
        Latitude of second point in degrees.
    lon2 : float
        Longitude of second point in degrees.
    radius_km : float, optional
        Spherical radius of the Earth in kilometers (default 6371.0).

    Returns
    -------
    float
        Great-circle distance in kilometers.
    """
    # Fast check for identical coordinates
    d_lat_deg = float(lat2) - float(lat1)
    d_lon_deg = (float(lon2) - float(lon1) + 180.0) % 360.0 - 180.0

    if abs(d_lat_deg) < 1e-12 and abs(d_lon_deg) < 1e-12:
        return 0.0

    # Clamp latitudes to physical boundaries
    lat1_c = max(-90.0, min(90.0, float(lat1)))
    lat2_c = max(-90.0, min(90.0, float(lat2)))

    # Recompute d_lat if clamping changed values
    d_lat_c = lat2_c - lat1_c

    phi1 = lat1_c * RAD_FACTOR
    phi2 = lat2_c * RAD_FACTOR

    sin_half_dphi = math.sin(d_lat_c * HALF_RAD_FACTOR)
    sin_half_dlambda = math.sin(d_lon_deg * HALF_RAD_FACTOR)

    a = (
        sin_half_dphi * sin_half_dphi
        + math.cos(phi1) * math.cos(phi2) * sin_half_dlambda * sin_half_dlambda
    )

    # Numerical safeguard: clamp a to [0.0, 1.0] to prevent math domain error in asin
    a = max(0.0, min(1.0, a))
    return (2.0 * radius_km) * math.asin(math.sqrt(a))


def compute_travel_speed(
    lat1: float,
    lon1: float,
    t1: float,
    lat2: float,
    lon2: float,
    t2: float,
    speed_threshold_kmh: float = IMPOSSIBLE_TRAVEL_SPEED_KMH,
    min_time_delta_hours: float = MIN_TIME_DELTA_HOURS,
) -> Tuple[float, int, float]:
    """
    Computes travel distance (km), velocity (km/h), and impossible travel jump flag.

    Formula:
        d = haversine_distance(lat1, lon1, lat2, lon2)
        delta_t_hours = max((t2 - t1) / 3600.0, min_time_delta_hours)
        v = d / delta_t_hours
        impossible = 1 if v > speed_threshold_kmh else 0

    Parameters
    ----------
    lat1, lon1, t1 : float
        Previous location (degrees) and epoch timestamp (seconds).
    lat2, lon2, t2 : float
        Current location (degrees) and epoch timestamp (seconds).
    speed_threshold_kmh : float, optional
        Speed threshold in km/h above which travel is considered impossible (default 800.0).
    min_time_delta_hours : float, optional
        Floor for time delta to safeguard against division by zero (default 1e-4 hours ~0.36s).

    Returns
    -------
    Tuple[float, int, float]
        (travel_speed_kmh, impossible_travel_flag, distance_km)
    """
    distance_km = haversine_distance(lat1, lon1, lat2, lon2)

    # If distance is zero, speed is zero regardless of timestamp delta
    if distance_km <= 0.0:
        return 0.0, 0, 0.0

    # Compute elapsed time in hours
    delta_t_sec = float(t2) - float(t1)
    delta_t_hours = max(delta_t_sec / 3600.0, min_time_delta_hours)

    travel_speed_kmh = distance_km / delta_t_hours
    impossible_flag = 1 if travel_speed_kmh > speed_threshold_kmh else 0

    return travel_speed_kmh, impossible_flag, distance_km
