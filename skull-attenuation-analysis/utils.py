import numpy as np
import pandas as pd


def compute_power_at_freq(t, v, freq_Hz, cutoff_freq):
    # Compute the power at f0 ± cutoff_freq using windowing
    f0 = freq_Hz
    fs = 1 / (t[1] - t[0])  # Sampling frequency

    # Apply Hann window
    window = np.hanning(len(v))
    v_windowed = v * window

    # Compute FFT
    n = len(v_windowed)
    fft_v = np.fft.fft(v_windowed)
    freqs = np.fft.fftfreq(n, 1 / fs)

    # Create a mask for frequencies within f0 ± cutoff_freq
    freq_mask = (freqs >= f0 - cutoff_freq) & (freqs <= f0 + cutoff_freq)

    # Compute power in the frequency band
    # Normalize by the sum of squared window values for correct scaling
    power = np.sum(np.abs(fft_v[freq_mask]) ** 2) / (np.sum(window**2) * fs)

    return power


def compute_power(file_num, tmin=1e-5, tmax=None, denoise=False, freq_Hz=None):
    filename = f"data/SDS{file_num:05d}.dat"
    try:
        df = pd.read_csv(filename, delimiter=",", header=None)
    except:
        print(f"Problem with file {filename}")
        return np.nan

    d = df.to_numpy()
    t = d[:, 0]
    v = d[:, 2]

    if tmax is None:
        tmax = max(t)

    t_mask = (t >= tmin) & (t <= tmax)
    t = t[t_mask]
    v = v[t_mask]

    if denoise:
        if freq_Hz is None:
            raise ValueError("Frequency must be provided if denoising")

        # pulse width is 1000 half-cycles
        ncycles = 500
        pulse_width_s = ncycles / freq_Hz
        cutoff_freq = 2 / pulse_width_s

        # restrict to pulse width data
        t_mask = t <= tmin + pulse_width_s
        t = t[t_mask]
        v = v[t_mask]

        power = compute_power_at_freq(t, v, freq_Hz, cutoff_freq)

    else:
        power = np.linalg.norm(v) ** 2
    # power = np.max(v)**2
    return power


# Function to check if a string can be converted to an integer
def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
