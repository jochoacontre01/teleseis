import numpy as np
from obspy.signal.rotate import rotate_ne_rt, rotate_zne_lqt


def nez_to_rtz(ncomp, ecomp, zcomp, baz):
    """
    NEZ_TO_RTZ(NCOMP, ECOMP, ZCOMP, BAZ)
    Rotates NS, EW, Z seismic records into Radial, Transverse and Z components.

    Inputs:
        ncomp, ecomp, zcomp : NS, EW, Z seismograms
        baz : Backazimuth in degrees, positive clockwise from North

    Outputs:
        rcomp : Radial component, with BAZ+180 positive
        tcomp : Transverse component, with BAZ+270 positive
        zcomp : Z component (untouched)
    """
    ncomp = np.atleast_2d(ncomp)
    ecomp = np.atleast_2d(ecomp)
    zcomp = np.atleast_2d(zcomp)
    baz = np.atleast_1d(baz)

    ntr, nsamp = ncomp.shape
    rcomp = np.zeros_like(ncomp)
    tcomp = np.zeros_like(ncomp)

    # Tile baz if only a single value is given for multiple traces
    if len(baz) == 1 and ntr > 1:
        baz = np.tile(baz, ntr)

    for ii in range(ntr):
        # ObsPy rotate_ne_rt returns R (baz) and T (baz+90).
        # MATLAB expects R to be BAZ+180 positive, and T to be BAZ+270 positive.
        # Therefore, we negate both R and T to match the original MATLAB output.
        r, t = rotate_ne_rt(ncomp[ii], ecomp[ii], baz[ii])
        rcomp[ii] = -r
        tcomp[ii] = -t

    if rcomp.shape[0] == 1:
        return rcomp[0], tcomp[0], zcomp[0]
    return rcomp, tcomp, zcomp


def nez_to_lqt(ncomp, ecomp, zcomp, baz, rayp, vp):
    """
    NEZ_TO_LQT(NCOMP, ECOMP, ZCOMP, BAZ, RAYP, VP)
    Rotate NS, EW, Z seismic records into L, Q, and T components.

    Inputs:
        ncomp, ecomp, zcomp : NS, EW, Z seismograms
        baz  : Backazimuth in degrees
        rayp : Ray parameter of incident wave in s/km
        vp   : P-wave velocity near the surface in km/s

    Outputs:
        lcomp : L component, parallel to incident P-wave, positive upward
        qcomp : Q component, perpendicular to incident P-wave, BAZ+180 positive
        tcomp : Transverse component, BAZ+270 positive
    """
    ncomp = np.atleast_2d(ncomp)
    ecomp = np.atleast_2d(ecomp)
    zcomp = np.atleast_2d(zcomp)
    baz = np.atleast_1d(baz)
    rayp = np.atleast_1d(rayp)
    vp = np.atleast_1d(vp)

    ntr, nsamp = ncomp.shape
    lcomp = np.zeros_like(ncomp)
    qcomp = np.zeros_like(ncomp)
    tcomp = np.zeros_like(ncomp)

    if len(baz) == 1 and ntr > 1:
        baz = np.tile(baz, ntr)
    if len(rayp) == 1 and ntr > 1:
        rayp = np.tile(rayp, ntr)
    if len(vp) == 1 and ntr > 1:
        vp = np.tile(vp, ntr)

    for ii in range(ntr):
        # Calculate incidence angle in degrees
        inc_angle = np.degrees(np.arcsin(rayp[ii] * vp[ii]))

        # obspy rotate_zne_lqt returns L, Q, T.
        # ObsPy's L points DOWN and towards the source. MATLAB's L is positive UPWARD.
        # ObsPy's Q points UP and AWAY from the source. MATLAB's Q is BAZ+180 positive (AWAY).
        # ObsPy's T points BAZ+90. MATLAB's T is BAZ+270.
        l, q, t = rotate_zne_lqt(zcomp[ii], ncomp[ii], ecomp[ii], baz[ii], inc_angle)

        lcomp[ii] = -l
        qcomp[ii] = q
        tcomp[ii] = -t

    if lcomp.shape[0] == 1:
        return lcomp[0], qcomp[0], tcomp[0]
    return lcomp, qcomp, tcomp


def nez_to_psvh(ncomp, ecomp, zcomp, baz, rayp, vp, vs):
    """
    NEZ_TO_PSVH(NCOMP, ECOMP, ZCOMP, BAZ, RAYP, VP, VS)
    Convert NS, EW, Z seismic records into P, SV, and SH components
    using the free surface transfer matrix.

    Outputs:
        pcomp  : P component, parallel to incident P-wave, positive upward
        svcomp : SV component, polarization direction of SV waves (BAZ+180 positive)
        shcomp : SH component, normal to vertical plane (BAZ+270 positive)
    """
    ncomp = np.atleast_2d(ncomp)
    ecomp = np.atleast_2d(ecomp)
    zcomp = np.atleast_2d(zcomp)
    baz = np.atleast_1d(baz)
    rayp = np.atleast_1d(rayp)
    vp = np.atleast_1d(vp)
    vs = np.atleast_1d(vs)

    ntr, nsamp = ncomp.shape
    pcomp = np.zeros_like(ncomp)
    svcomp = np.zeros_like(ncomp)
    shcomp = np.zeros_like(ncomp)

    if len(baz) == 1 and ntr > 1:
        baz = np.tile(baz, ntr)
    if len(rayp) == 1 and ntr > 1:
        rayp = np.tile(rayp, ntr)
    if len(vp) == 1 and ntr > 1:
        vp = np.tile(vp, ntr)
    if len(vs) == 1 and ntr > 1:
        vs = np.tile(vs, ntr)

    for ii in range(ntr):
        # 1. Rotate to Radial/Transverse (Mapped to MATLAB's BAZ conventions)
        r_obs, t_obs = rotate_ne_rt(ncomp[ii], ecomp[ii], baz[ii])
        r = -r_obs
        t = -t_obs

        shcomp[ii] = t  # SH is identical to Transverse

        # 2. Free surface transfer matrix for P and SV
        p = rayp[ii]
        alpha = vp[ii]
        beta = vs[ii]

        # Vertical slowness
        qa = np.sqrt(1.0 / alpha**2 - p**2)
        qb = np.sqrt(1.0 / beta**2 - p**2)

        # Apply transformation matrix to isolate upgoing P and SV waves
        # Assuming Z is positive UP, R is positive AWAY (baz+180)
        term = 1.0 - 2.0 * beta**2 * p**2

        pcomp[ii] = (term / (2.0 * alpha * qa)) * zcomp[ii] - (p * beta**2 / alpha) * r
        svcomp[ii] = (p * beta) * zcomp[ii] + (term / (2.0 * beta * qb)) * r

    if pcomp.shape[0] == 1:
        return pcomp[0], svcomp[0], shcomp[0]
    return pcomp, svcomp, shcomp
