from scipy.signal import butter, filtfilt
import numpy as np
import numpy.typing as npt

def bpfilt(x: npt.ArrayLike, dt: float, lf: float, hf: float) -> npt.NDArray:
    """
    Apply a zero-phase bandpass Butterworth filter to a signal.

    Parameters
    ----------
    x : array_like
        Input signal(s). Can be a 1D array or a 2D array where traces are rows.
    dt : float
        Sample interval in seconds.
    lf : float
        Low cutoff frequency in Hz.
    hf : float
        High cutoff frequency in Hz.

    Returns
    -------
    y : ndarray
        Filtered signal(s), same shape as `x`.
    """
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

def decon(fseis: npt.ArrayLike, source: npt.ArrayLike, water: float, dt: float = 0.05, tshift: float = 10.0) -> npt.NDArray:
    """
    Deconvolve `source` from `fseis`, with a water level (damping factor).
    
    Inputs can take three configurations:
    (1) `fseis` and `source` are single traces, producing a single deconvolved trace.
    (2) `fseis` is a 2D array with traces forming rows, and `source` is a single trace, 
        where each trace of `fseis` is deconvolved by the single source.
    (3) `fseis` and `source` are 2D arrays with traces forming rows, where each trace 
        in `fseis` is deconvolved by each corresponding source in `source`.

    Parameters
    ----------
    fseis : array_like
        Input trace(s) to be deconvolved.
    source : array_like
        Source trace(s) or wavelet(s) used for deconvolution.
    water : float
        Water level (damping factor) to stabilize the denominator.
    dt : float, optional
        Sample interval in seconds. Default is 0.05.
    tshift : float, optional
        Time shift to apply to the deconvolved signal in seconds. Default is 10.0.

    Returns
    -------
    dseis : ndarray
        Deconvolved signal(s). Returns a 1D array if input was 1D, or a 
        2D array if input was 2D.
    """
    # Standardize inputs to 2D arrays
    fseis_arr = np.asarray(fseis)
    source_arr = np.asarray(source)
    is_1d = fseis_arr.ndim == 1 or min(fseis_arr.shape) == 1
    
    if is_1d:
        if fseis_arr.ndim == 1:
            fseis_arr = fseis_arr.reshape(1, -1)
        else:
            fseis_arr = fseis_arr.flatten().reshape(1, -1)
            
    if source_arr.ndim == 1:
        source_arr = source_arr.reshape(1, -1)
        
    slength = fseis_arr.shape[1]
    n2 = 2 ** int(np.ceil(np.log2(slength)))
    
    fseisft = np.fft.rfft(fseis_arr, n=n2, axis=1)
    sourceft = np.fft.rfft(source_arr, n=n2, axis=1)
    
    power_spec = np.real(sourceft * np.conj(sourceft))
    
    freq = np.fft.rfftfreq(n2, d=dt)
    wtshift = -1j * 2.0 * np.pi * freq * tshift
    
    # Deconvolution operator with water level stabilizer
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

def taper(x: npt.ArrayLike, nt: float, dt: float, t1: float, t2: float) -> npt.NDArray:
    """
    Apply a cosine-bell taper to the ends of a signal.

    Parameters
    ----------
    x : array_like
        Input signal(s). Can be a 1D array or a 2D array where traces are rows.
    nt : float
        Duration of the tapering edge in seconds.
    dt : float
        Sample interval in seconds.
    t1 : float
        Start time of the full amplitude window in seconds.
    t2 : float
        End time of the full amplitude window in seconds.

    Returns
    -------
    y : ndarray
        The tapered signal(s).
    """
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

    # Apply leading taper and zero out prior
    if it1 > 0:
        taper_arr[:it1] = 0.0
    if it1 + it_len + 1 <= nx:
        taper_arr[it1:it1+it_len+1] = ct
    else:
        taper_arr[it1:] = ct[:nx-it1]

    # Apply trailing taper and zero out after
    if it2 < nx:
        taper_arr[it2+1:] = 0.0
    
    start_back = max(0, it2 - it_len)
    back_len = it2 - start_back + 1
    if back_len > 0:
        taper_arr[start_back:it2+1] = ct[::-1][-back_len:]

    # Apply taper array to each trace
    y = x_arr * taper_arr
    
    return y[0] if is_1d else y