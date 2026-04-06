import numpy as np
from scipy.signal import detrend
from spectral import taper, bpfilt
import matplotlib.pyplot as plt

def powspec(seis, dt):
    """
    POWSPEC calculates and plots the power spectral density of seismogram(s) in SEIS. 
    
    DT is the sample interval and SPEC is the power spectrum. If SEIS is a single 
    seismogram, then a log-log plot of the power spectrum is displayed. If SEIS 
    is a matrix, then the spectrum is calculated for each line of SEIS, but no 
    plot is produced.
    """
    seis_arr = np.asarray(seis)
    is_1d = seis_arr.ndim == 1 or min(seis_arr.shape) == 1
    
    # Ensure it's treated as a 2D array with traces along rows
    if is_1d:
        if seis_arr.ndim == 1:
            seis_2d = seis_arr.reshape(1, -1)
        else:
            # Flatten to 1D then reshape to guarantee row vector
            seis_2d = seis_arr.flatten().reshape(1, -1)
    else:
        seis_2d = seis_arr
        
    # Calculate FFT along the rows (axis=1 or -1)
    fftseis = np.fft.fft(seis_2d, axis=1)
    
    # Calculate amplitude spectrum 
    # MATLAB uses: (real(fftseis).^2+imag(fftseis).^2).^.5 which is equivalent to np.abs
    spec = np.abs(fftseis)
    
    # Plotting condition: min(size(spec)) == 1
    if is_1d:
        plt.figure(figsize=(8, 6))
        lspec = spec.shape[1]
        lhspec = int(np.round(lspec / 2))
        frequ = 1.0 / dt
        nyfrequ = frequ / 2.0
        minfrequ = 1.0 / (lspec * dt)
        
        # Emulate MATLAB's array creation minfrequ:nyfrequ/lhspec:nyfrequ
        step = nyfrequ / lhspec
        freq_axis = np.arange(minfrequ, nyfrequ + step*0.5, step)
        
        # Guard against minor array length differences caused by floats
        plot_len = min(len(freq_axis), lhspec)
        
        plt.loglog(freq_axis[:plot_len], spec[0, :plot_len])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Power Spectrum')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.show()
        
    # Reshape back to original dimensions if needed
    if seis_arr.ndim == 1:
        return spec[0]
    elif seis_arr.shape[0] > seis_arr.shape[1]:
        return spec.T # return column vector if input was column vector
    else:
        return spec
    
def plot_sectiond(seis, aflag, delta=None, lf=None, hf=None, 
                  title_str="Seismogram Section", xlabel_str="Trace Number"):
    """
    Plots a seismogram section as a function of depth.
    
    Parameters:
        seis (2D array): Array of seismograms (traces forming rows).
        aflag (float): Determines amplitude scaling. 
                       < 0: Trace scaled to its maximum.
                       == 0: Trace scaled to max of entire section.
                       > 0: Trace scaled to aflag value.
        delta (list/array): Position of geophones. Can be None/empty.
        lf, hf (float): Low and high cutoff frequencies for bandpass. Can be None.
        title_str, xlabel_str (str): Plot labels (replacing MATLAB's inputname).
    """
    # Convert input to float numpy array to prevent integer division issues
    seis = np.array(seis, dtype=float)
    
    # Hardwired variables (begin time and time step) 
    beg = -5.0
    dt = 0.1

    # Flip polarity
    seis = -seis

    # Measuring size of seis
    ny, nt = seis.shape

    # Default spacing between traces
    if delta is None or len(delta) == 0:
        dtest = 1
        delta = np.arange(1, ny + 1)
    else:
        dtest = 0
        delta = np.array(delta)

    # Setting scale for each trace
    sdel = np.sort(delta)
    if len(sdel) > 1:
        ddelta = max((sdel[-1] - sdel[0]) / len(sdel), np.min(np.diff(sdel)) * 2)
    else:
        ddelta = 1.0 # Fallback if only one trace is provided
        
    wb = ddelta / 20.0

    # Precondition seis
    for iy in range(ny):
        seis[iy, :] = seis[iy, :] - np.mean(seis[iy, :])
        seis[iy, :] = detrend(seis[iy, :])
        seis[iy, :] = taper(seis[iy, :], 0.2, dt, 0.6, nt * dt - 0.6)

    # Apply bandpass filter if frequencies are provided
    if lf is not None and hf is not None:
        seis = bpfilt(seis, dt, lf, hf)

    # Precondition again after filtering
    for iy in range(ny):
        seis[iy, :] = seis[iy, :] - np.mean(seis[iy, :])
        seis[iy, :] = detrend(seis[iy, :])
        seis[iy, :] = taper(seis[iy, :], 0.2, dt, 0.6, nt * dt - 0.6)

    # Initialize coordinate matrices
    xymat = np.zeros_like(seis)
    bxymat = np.zeros_like(seis)
    rxymat = np.zeros_like(seis)

    # Calculate coordinates for plotting and shading based on aflag
    if aflag < 0:
        for iy in range(ny):
            normf = np.max(np.abs(seis[iy, :])) + 0.0000001
            xymat[iy, :] = delta[iy] - ddelta * seis[iy, :] / normf
            
            val_b = seis[iy, :] - wb / ddelta * normf
            bxymat[iy, :] = delta[iy] - wb - ddelta * ((np.sign(val_b) + 1) / 2) * val_b / normf
            
            val_r = -seis[iy, :] - wb / ddelta * normf
            rxymat[iy, :] = delta[iy] + wb - ddelta * ((np.sign(val_r) + 1) / 2) * (seis[iy, :] + wb / ddelta * normf) / normf

    elif aflag == 0:
        normf = np.max(np.abs(seis)) + 0.0000001
        for iy in range(ny):
            xymat[iy, :] = delta[iy] - ddelta * seis[iy, :] / normf
            
            val_b = seis[iy, :] - wb / ddelta * normf
            bxymat[iy, :] = delta[iy] - wb - ddelta * ((np.sign(val_b) + 1) / 2) * val_b / normf
            
            val_r = -seis[iy, :] - wb / ddelta * normf
            rxymat[iy, :] = delta[iy] + wb - ddelta * ((np.sign(val_r) + 1) / 2) * (seis[iy, :] + wb / ddelta * normf) / normf
            
            # MATLAB debugging pause block (Commented out for smooth Python execution)
            # print(np.column_stack((xymat[iy, :100], bxymat[iy, :100], rxymat[iy, :100])))
            # input("Press Enter to continue...")

    else:
        for iy in range(ny):
            xymat[iy, :] = delta[iy] - ddelta * seis[iy, :] / aflag
            
            val_b = seis[iy, :] - wb / ddelta * aflag
            bxymat[iy, :] = delta[iy] - wb - ddelta * ((np.sign(val_b) + 1) / 2) * val_b / aflag
            
            val_r = -seis[iy, :] - wb / ddelta * aflag
            rxymat[iy, :] = delta[iy] + wb - ddelta * ((np.sign(val_r) + 1) / 2) * (seis[iy, :] + wb / ddelta * aflag) / aflag

    # Plotting
    time = np.arange(nt) * dt + beg
    fig, ax = plt.subplots(figsize=(10, 8))

    for iy in range(ny):
        # Plot base waveform
        ax.plot(xymat[iy, :], time, 'k-')
        
        # Fill blue and red polygons (Edgecolor mapped from [1 1 1] to white)
        ax.fill(bxymat[iy, :], time, 'b', edgecolor='white', linewidth=0.5)
        ax.fill(rxymat[iy, :], time, 'r', edgecolor='white', linewidth=0.5)

    # Plot the black line one more time to overlay it on top of the fills
    for iy in range(ny):
        ax.plot(xymat[iy, :], time, 'k-', linewidth=1)

    # Format Axes
    ax.set_ylabel('Depth [km]')
    ax.set_title(title_str, fontsize=14)
    
    if dtest == 1:
        ax.set_xlabel('Trace number')
    else:
        ax.set_xlabel(xlabel_str)

    # Reverse Y-Axis (Depth) and tighten boundaries
    ax.invert_yaxis()
    ax.autoscale(enable=True, axis='both', tight=True)

    plt.show()

def plot_traces(*traces, labels=None):
    """
    PLOT_TRACES(TRACE1, TRACE2, TRACE3, ...)
    Plot time series TRACE1, TRACE2, TRACE3, ... where TRACE[N] is a row
    or column of numbers. By default, traces are plotted with vertical
    axis between -1 and 1. To optimize vertical axis for individual
    traces, uncomment command line "axis tight" in the code.
    
    Python adaptation: Pass `labels=['trace1', 'trace2']` as a kwarg since 
    Python cannot implicitly extract input variable names like MATLAB.
    """
    shift = 10.0
    dt = 0.05
    ntr = len(traces)
    
    if ntr == 0:
        return
        
    fig, axs = plt.subplots(ntr, 1, figsize=(10, 2.5 * ntr))
    if ntr == 1:
        axs = [axs]
        
    for ii, trace in enumerate(traces):
        trace_arr = np.asarray(trace).flatten()
        time = np.arange(1, len(trace_arr) + 1) * dt - shift
        
        axs[ii].plot(time, trace_arr)
        axs[ii].set_xlim([0 - shift, len(trace_arr) * dt - shift])
        axs[ii].set_ylim([-1, 1])
        
        # To optimize the vertical axis for each trace, uncomment line below:
        # axs[ii].autoscale(enable=True, axis='both', tight=True)
        
        lbl = labels[ii] if labels and ii < len(labels) else f'Trace {ii+1}'
        axs[ii].legend([lbl], loc='upper right')
        
    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def map_1rf(rfun, rayp):
    """
    MAP_1RF(RFUN, RAYP)
    Maps a single receiver function RFUN with ray parameter RAYP (s/km) 
    to depth by 1D migration
    """
    tshift = 10.0
    dt = 0.05
    
    rfun_arr = np.asarray(rfun).flatten()
    split_idx = int(np.fix(tshift / dt)) - 1
    
    prfun = rfun_arr[:split_idx]
    main_rfun = rfun_arr[split_idx:]
    
    # Velocity model (note that this is not a perfectly realistic model)
    #       d1 (km)   d2 (km)   Vp (km/s)   Vs (km/s)   rho (g/cm^3)
    vmod = np.array([
        [ 0,         2,         3,           1.5,         2.28 ],
        [ 2,        37,         6,           3.4,         2.6  ],
        [37,      1000,         8.1,         4.5,         3.5  ]
    ])
    
    d1 = vmod[:, 0]
    d2 = vmod[:, 1] + 0.001
    vp = vmod[:, 2]
    vs = vmod[:, 3]
    
    # Compute depths for conversion for (1) Ps, (2) Pps, and (3) Pss waves
    n_layers = len(d2)
    tcut1 = np.zeros(n_layers + 1)
    tcut2 = np.zeros(n_layers + 1)
    tcut3 = np.zeros(n_layers + 1)
    
    # Determine cutoff times at each of the model's interfaces
    for ii in range(1, n_layers + 1):
        idx = ii - 1
        dz = d2[idx] - d1[idx]
        term_s = np.sqrt(1 / vs[idx]**2 - rayp**2)
        term_p = np.sqrt(1 / vp[idx]**2 - rayp**2)
        
        tcut1[ii] = (term_s - term_p) * dz + tcut1[ii - 1]
        tcut2[ii] = (term_s + term_p) * dz + tcut2[ii - 1]
        tcut3[ii] = (2 * term_s) * dz + tcut3[ii - 1]
        
    n_main = len(main_rfun)
    depth1 = np.zeros(n_main)
    depth2 = np.zeros(n_main)
    depth3 = np.zeros(n_main)
    
    # Loop over times of the RF and find depth
    for it in range(n_main):
        rftime = (it + 1) * dt
        
        tt1 = np.where(tcut1 < rftime)[0]
        tt2 = np.where(tcut2 < rftime)[0]
        tt3 = np.where(tcut3 < rftime)[0]
        
        m1 = tt1[-1] if len(tt1) > 0 else 0
        m2 = tt2[-1] if len(tt2) > 0 else 0
        m3 = tt3[-1] if len(tt3) > 0 else 0
        
        depth1[it] = d1[m1] + (rftime - tcut1[m1]) / (np.sqrt(1/vs[m1]**2 - rayp**2) - np.sqrt(1/vp[m1]**2 - rayp**2))
        depth2[it] = d1[m2] + (rftime - tcut2[m2]) / (np.sqrt(1/vs[m2]**2 - rayp**2) + np.sqrt(1/vp[m2]**2 - rayp**2))
        depth3[it] = d1[m3] + (rftime - tcut3[m3]) / (2 * np.sqrt(1/vs[m3]**2 - rayp**2))

    # Plot results
    dd = np.arange(-5, 100.1, 0.1)
    
    def build_interp(depth_arr, main_rf, prf):
        idx = np.where(depth_arr <= 110)[0]
        idxp = np.where(depth_arr <= 10)[0]
        num_p = len(idxp)
        
        xp = np.concatenate([-np.flip(depth_arr[idxp]), [0], depth_arr[idx[:-1]]])
        yp_presig = prf[-num_p:] if num_p > 0 else []
        yp = np.concatenate([yp_presig, main_rf[idx]])
        
        # Sort arrays because np.interp requires strictly monotonically increasing x values
        sort_indices = np.argsort(xp)
        return np.interp(dd, xp[sort_indices], yp[sort_indices])

    rfun1 = build_interp(depth1, main_rfun, prfun)
    rfun2 = build_interp(depth2, main_rfun, prfun)
    rfun3 = build_interp(depth3, main_rfun, prfun)

    plt.figure(figsize=(10, 4))
    plt.plot(dd, rfun1, label='Ps')
    plt.legend(loc='upper right')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.title('Forward (standard) receiver function')
    plt.xlabel('Depth (km)')
    plt.show(block=False)
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    
    axs[0].plot(dd, rfun1, label='Ps')
    axs[0].legend(loc='upper right')
    axs[0].autoscale(enable=True, axis='both', tight=True)
    axs[0].set_title('Forward and reflected (multiple-based) receiver functions')
    
    axs[1].plot(dd, rfun2, label='Pps')
    axs[1].legend(loc='upper right')
    axs[1].autoscale(enable=True, axis='both', tight=True)
    
    axs[2].plot(dd, -rfun3, label='Pss')
    axs[2].legend(loc='upper right')
    axs[2].autoscale(enable=True, axis='both', tight=True)
    
    # Stack forward and multiples
    axs[3].plot(dd, rfun1 + rfun2 - rfun3, label='Sum of modes')
    axs[3].legend(loc='upper right')
    axs[3].autoscale(enable=True, axis='both', tight=True)
    axs[3].set_xlabel('Depth (km)')
    
    plt.tight_layout()
    plt.show()
    
def compare_traces(*traces, labels=None):
    """
    COMPARE_TRACES(TRACE1, TRACE2, TRACE3, ...)
    Plot time series TRACE1, TRACE2, TRACE3, ... in the same graph
    for comparison, where TRACE[N] is a row or column of numbers.  By
    default, traces are plotted with vertical axis between -1 and 1.
    To optimize vertical axis for individual traces, uncomment
    command line "axis tight" in the code.
    """
    shift = 10.0
    dt = 0.05
    ntr = len(traces)
    
    plt.figure(figsize=(10, 4))
    
    for ii in range(ntr):
        trace = np.asarray(traces[ii]).flatten()
        time_axis = np.arange(1, len(trace) + 1) * dt - shift
        
        lbl = labels[ii] if labels and ii < len(labels) else f'Trace {ii+1}'
        plt.plot(time_axis, trace, label=lbl)
        
    plt.xlim([0 - shift, len(trace) * dt - shift])
    plt.ylim([-1, 1])
    
    # To optimize the vertical axis for each trace, uncomment line below:
    plt.autoscale(enable=True, axis='y', tight=True)
    
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()