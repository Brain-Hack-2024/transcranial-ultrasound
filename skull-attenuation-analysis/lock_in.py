import numpy as np
from scipy.signal import butter, filtfilt


def generate_reference_signal(freq, t):
    """
    Generates a reference sine wave signal.

    Parameters:
        freq: Frequency of the reference signal.
        t: Time array.

    Returns:
        Reference sine wave signal.
    """
    return np.sin(2 * np.pi * freq * t)


def low_pass_filter(signal, cutoff_freq, sample_rate, order=5):
    """
    Applies a low-pass Butterworth filter to the signal.

    Parameters:
        signal: The signal to be filtered.
        cutoff_freq: The cutoff frequency of the low-pass filter.
        sample_rate: The sample rate of the signal.
        order: The order of the Butterworth filter (default is 5).

    Returns:
        Filtered signal.
    """
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def lock_in_amplifier(input_signal, reference_signal, cutoff_freq, sample_rate):
    """
    Implements a lock-in amplifier.

    Parameters:
        input_signal: The noisy input signal.
        reference_signal: The reference sine wave signal.
        cutoff_freq: The cutoff frequency of the low-pass filter.
        sample_rate: The sample rate of the input signal.

    Returns:
        The output of the lock-in amplifier (filtered signal).
    """
    multiplied_signal = input_signal * reference_signal
    output_signal = low_pass_filter(multiplied_signal, cutoff_freq, sample_rate)
    return output_signal


# Example Usage
sample_rate = 10000  # Sample rate in Hz
t = np.linspace(0, 1.0, sample_rate)  # Time array for 1 second

# Parameters
input_freq = 50  # Frequency of the signal of interest in Hz
reference_freq = 50  # Same as input frequency
noise_level = 0.5  # Noise level
low_pass_cutoff = 10  # Cutoff frequency for the low-pass filter in Hz

# Generate input signal (signal of interest + noise)
signal_of_interest = np.sin(2 * np.pi * input_freq * t)
noise = noise_level * np.random.randn(len(t))
input_signal = signal_of_interest + noise

# Generate reference signal
reference_signal = generate_reference_signal(reference_freq, t)

# Apply lock-in amplifier
output_signal = lock_in_amplifier(
    input_signal, reference_signal, low_pass_cutoff, sample_rate
)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, input_signal)
plt.title("Input Signal (with Noise)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(t, reference_signal)
plt.title("Reference Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(t, output_signal)
plt.title("Output Signal (after Lock-In Amplifier)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
