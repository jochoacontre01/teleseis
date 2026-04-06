import numpy as np
import numpy.typing as npt
from obspy.signal.rotate import rotate_ne_rt, rotate_zne_lqt


def nez_to_rtz(
    ncomp: npt.ArrayLike, 
    ecomp: npt.ArrayLike, 
    zcomp: npt.ArrayLike, 
    baz: npt.ArrayLike | float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Rotates NS, EW, Z seismic records into Radial, Transverse and Z components.

    Parameters
    ----------
    ncomp : array_like
        North-South seismogram(s).
    ecomp : array_like
        East-West seismogram(s).
    zcomp : array_like
        Vertical (Z) seismogram(s).
    baz : array_like or float
        Backazimuth in degrees, positive clockwise from North.

    Returns
    -------
    rcomp : ndarray
        Radial component, with BAZ+180 positive.
    tcomp : ndarray
        Transverse component, with BAZ+270 positive.
    zcomp : ndarray
        Z component (untouched).
    """
    ncomp_arr = np.atleast_2d(ncomp)
    ecomp_arr = np.atleast_2d(ecomp)
    zcomp_arr = np.atleast_2d(zcomp)
    baz_arr = np.atleast_1d(baz)

    ntr, _ = ncomp_arr.shape
    rcomp = np.zeros_like(ncomp_arr)
    tcomp = np.zeros_like(ncomp_arr)

    # Tile baz if only a single value is given for multiple traces
    if len(baz_arr) == 1 and ntr > 1:
        baz_arr = np.tile(baz_arr, ntr)

    for ii in range(ntr):
        # ObsPy rotate_ne_rt returns R (baz) and T (baz+90).
        # MATLAB expects R to be BAZ+180 positive, and T to be BAZ+270 positive.
        # Therefore, we negate both R and T to match the original MATLAB output.
        r, t = rotate_ne_rt(ncomp_arr[ii], ecomp_arr[ii], baz_arr[ii])
        rcomp[ii] = -r
        tcomp[ii] = -t

    if rcomp.shape[0] == 1:
        return rcomp[0], tcomp[0], zcomp_arr[0]
    return rcomp, tcomp, zcomp_arr


def nez_to_lqt(
    ncomp: npt.ArrayLike, 
    ecomp: npt.ArrayLike, 
    zcomp: npt.ArrayLike, 
    baz: npt.ArrayLike | float, 
    rayp: npt.ArrayLike | float, 
    vp: npt.ArrayLike | float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Rotate NS, EW, Z seismic records into L, Q, and T components.

    Parameters
    ----------
    ncomp : array_like
        North-South seismogram(s).
    ecomp : array_like
        East-West seismogram(s).
    zcomp : array_like
        Vertical (Z) seismogram(s).
    baz : array_like or float
        Backazimuth in degrees.
    rayp : array_like or float
        Ray parameter of incident wave in s/km.
    vp : array_like or float
        P-wave velocity near the surface in km/s.

    Returns
    -------
    lcomp : ndarray
        L component, parallel to incident P-wave, positive upward.
    qcomp : ndarray
        Q component, perpendicular to incident P-wave, BAZ+180 positive.
    tcomp : ndarray
        Transverse component, BAZ+270 positive.
    """
    ncomp_arr = np.atleast_2d(ncomp)
    ecomp_arr = np.atleast_2d(ecomp)
    zcomp_arr = np.atleast_2d(zcomp)
    baz_arr = np.atleast_1d(baz)
    rayp_arr = np.atleast_1d(rayp)
    vp_arr = np.atleast_1d(vp)

    ntr, _ = ncomp_arr.shape
    lcomp = np.zeros_like(ncomp_arr)
    qcomp = np.zeros_like(ncomp_arr)
    tcomp = np.zeros_like(ncomp_arr)

    if len(baz_arr) == 1 and ntr > 1:
        baz_arr = np.tile(baz_arr, ntr)
    if len(rayp_arr) == 1 and ntr > 1:
        rayp_arr = np.tile(rayp_arr, ntr)
    if len(vp_arr) == 1 and ntr > 1:
        vp_arr = np.tile(vp_arr, ntr)

    for ii in range(ntr):
        # Calculate incidence angle in degrees
        inc_angle = np.degrees(np.arcsin(rayp_arr[ii] * vp_arr[ii]))

        # obspy rotate_zne_lqt returns L, Q, T.
        # ObsPy's L points DOWN and towards the source. MATLAB's L is positive UPWARD.
        # ObsPy's Q points UP and AWAY from the source. MATLAB's Q is BAZ+180 positive (AWAY).
        # ObsPy's T points BAZ+90. MATLAB's T is BAZ+270.
        l, q, t = rotate_zne_lqt(zcomp_arr[ii], ncomp_arr[ii], ecomp_arr[ii], baz_arr[ii], inc_angle)

        lcomp[ii] = -l
        qcomp[ii] = q
        tcomp[ii] = -t

    if lcomp.shape[0] == 1:
        return lcomp[0], qcomp[0], tcomp[0]
    return lcomp, qcomp, tcomp


def nez_to_psvh(
    ncomp: npt.ArrayLike, 
    ecomp: npt.ArrayLike, 
    zcomp: npt.ArrayLike, 
    baz: npt.ArrayLike | float, 
    rayp: npt.ArrayLike | float, 
    vp: npt.ArrayLike | float, 
    vs: npt.ArrayLike | float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Convert NS, EW, Z seismic records into P, SV, and SH components
    using the free surface transfer matrix.

    Parameters
    ----------
    ncomp : array_like
        North-South seismogram(s).
    ecomp : array_like
        East-West seismogram(s).
    zcomp : array_like
        Vertical (Z) seismogram(s).
    baz : array_like or float
        Backazimuth in degrees.
    rayp : array_like or float
        Ray parameter of incident wave in s/km.
    vp : array_like or float
        P-wave velocity near the surface in km/s.
    vs : array_like or float
        S-wave velocity near the surface in km/s.

    Returns
    -------
    pcomp : ndarray
        P component, parallel to incident P-wave, positive upward.
    svcomp : ndarray
        SV component, polarization direction of SV waves (BAZ+180 positive).
    shcomp : ndarray
        SH component, normal to vertical plane (BAZ+270 positive).
    """
    ncomp_arr = np.atleast_2d(ncomp)
    ecomp_arr = np.atleast_2d(ecomp)
    zcomp_arr = np.atleast_2d(zcomp)
    baz_arr = np.atleast_1d(baz)
    rayp_arr = np.atleast_1d(rayp)
    vp_arr = np.atleast_1d(vp)
    vs_arr = np.atleast_1d(vs)

    ntr, _ = ncomp_arr.shape
    pcomp = np.zeros_like(ncomp_arr)
    svcomp = np.zeros_like(ncomp_arr)
    shcomp = np.zeros_like(ncomp_arr)

    if len(baz_arr) == 1 and ntr > 1:
        baz_arr = np.tile(baz_arr, ntr)
    if len(rayp_arr) == 1 and ntr > 1:
        rayp_arr = np.tile(rayp_arr, ntr)
    if len(vp_arr) == 1 and ntr > 1:
        vp_arr = np.tile(vp_arr, ntr)
    if len(vs_arr) == 1 and ntr > 1:
        vs_arr = np.tile(vs_arr, ntr)

    for ii in range(ntr):
        # 1. Rotate to Radial/Transverse (Mapped to MATLAB's BAZ conventions)
        r_obs, t_obs = rotate_ne_rt(ncomp_arr[ii], ecomp_arr[ii], baz_arr[ii])
        r = -r_obs
        t = -t_obs

        shcomp[ii] = 0.5 * t  # SH is identical to Transverse

        # 2. Free surface transfer matrix for P and SV
        p = rayp_arr[ii]
        alpha = vp_arr[ii]
        beta = vs_arr[ii]

        # Vertical slowness
        qa = np.sqrt(1.0 / alpha**2 - p**2)
        qb = np.sqrt(1.0 / beta**2 - p**2)

        # Apply transformation matrix to isolate upgoing P and SV waves
        # Assuming Z is positive UP, R is positive AWAY (baz+180)
        term = 1.0 - 2.0 * beta**2 * p**2

        pcomp[ii] = (term / (2.0 * alpha * qa)) * zcomp_arr[ii] + (p * beta**2 / alpha) * r
        svcomp[ii] = - (p * beta) * zcomp_arr[ii] + (term / (2.0 * beta * qb)) * r

    if pcomp.shape[0] == 1:
        return pcomp[0], svcomp[0], shcomp[0]
    return pcomp, svcomp, shcomp