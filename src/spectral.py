from scipy.signal import butter, filtfilt
import numpy as np

def bpfilt(x, dt, lf, hf):
    nyq = 0.5 / dt
    low = lf / nyq
    high = hf / nyq
    b, a = butter(2, [low, high], btype='band')
    
    x_arr = np.asarray(x)
    if x_arr.ndim == 1:
        return filtfilt(b, a, x_arr)
    else:
        y = np.zeros_like(x_arr)
        for ix in range(x_arr.shape[0]):
            y[ix, :] = filtfilt(b, a, x_arr[ix, :])
        return y

def decon(fseis, source, water):
    """
    DECON deconvolves SOURCE from FSEIS, with water level (damping factor) WATER.
    
    Inputs can take three configurations:
    (1) FSEIS and SOURCE are single traces, producing a single deconvolved trace; 
    (2) FSEIS is a 2-D array with traces forming lines, and SOURCE is a single trace, 
        where each trace of FSEIS is deconvolved by the single source in SOURCE; 
    (3) FSEIS and SOURCE are 2-D arrays with traces forming lines, where each trace 
        in FSEIS is deconvolved by each corresponding source in SOURCE.
    """
    # Standardize inputs to 2D arrays
    fseis_arr = np.atleast_2d(fseis)
    source_arr = np.atleast_2d(source)
    
    is_1d = np.asarray(fseis).ndim == 1
    ndt = 0.05
    
    ntr, slength = fseis_arr.shape
    nsrc = source_arr.shape[0]
    
    # Verify that input traces have the same duration
    if source_arr.shape[1] != slength:
        raise ValueError("The two input files should have the same length ... ABORT!")
        
    # Verify the size of source array. If it contains only one trace, 
    # repeat trace to match the number of traces in fseis
    if nsrc == 1:
        source_arr = np.tile(source_arr, (ntr, 1))
        
    # Calculate next power of 2 for FFT
    n2 = int(2 ** np.ceil(np.log2(slength)))
    
    tshift = 10.0
    omega = np.arange(n2 // 2 + 1) * 2 * np.pi / (n2 * ndt)
    wtshift = -1j * omega * tshift
    
    # Calculate FFT (along the rows)
    fseisft = np.fft.fft(fseis_arr, n=n2, axis=1)
    sourceft = np.fft.fft(source_arr, n=n2, axis=1)
    
    # Truncate up to Nyquist frequency
    sourceft = sourceft[:, :n2 // 2 + 1]
    fseisft = fseisft[:, :n2 // 2 + 1]
    
    # Water level deconvolution with damping factor
    # sourceft * conj(sourceft) gives the real power spectrum
    power_spec = np.real(sourceft * np.conj(sourceft))
    dseisft = (fseisft * np.conj(sourceft)) / (power_spec + water)
    
    # Apply time shift
    dseisft = dseisft * np.exp(wtshift)
    
    # Inverse FFT. Using irfft automatically assumes the input is the positive 
    # frequency half of a Hermitian spectrum (including DC and Nyquist), 
    # and reconstructs the full real time-domain signal.
    dseis = np.fft.irfft(dseisft, n=n2, axis=1)
    
    # Truncate to the original length
    dseis = dseis[:, :slength]
    
    return dseis[0] if is_1d else dseis

def taper(x, nt, dt, t1, t2):
    x_arr = np.asarray(x)
    is_1d = x_arr.ndim == 1
    if is_1d:
        x_arr = x_arr.reshape(1, -1)
        
    ns, nx = x_arr.shape
    taper_arr = np.ones(nx)
    
    it_len = int(np.fix(nt / dt))
    it = np.arange(it_len + 1) * dt / nt
    ct = 0.5 - 0.5 * np.cos(np.pi * it)
    
    it1 = int(np.fix(t1 / dt))
    it2 = int(np.fix(t2 / dt))
    
    if t1 > 0:
        start_idx = it1 - it_len
        taper_arr[start_idx : it1 + 1] = ct
        taper_arr[:start_idx] = 0
        
    if t2 > 0:
        if t2 > nx * dt - nt:
            raise ValueError(f"T2 > NX * DT - NT: {t2} > {nx*dt - nt}")
        taper_arr[it2 : it2 + it_len + 1] = ct[::-1]
        taper_arr[it2 + it_len :] = 0
        
    y = np.zeros_like(x_arr)
    for is_idx in range(ns):
        y[is_idx, :] = taper_arr * x_arr[is_idx, :]
        
    return y[0] if is_1d else y
