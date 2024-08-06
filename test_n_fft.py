import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
sampling_rate = 160  # Sampling rate in Hz
t = np.linspace(0, 1, sampling_rate, endpoint=False)  # 1 second time vector

# Generate a signal with multiple frequencies
frequencies = [5, 20, 40]  # Frequencies in Hz
signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies)

# Plot the original signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Function to plot FFT with different number of points
def plot_fft(signal, sampling_rate, num_points):
    fft_result = np.fft.fft(signal, num_points)
    fft_freqs = np.fft.fftfreq(num_points, 1/sampling_rate)
    
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freqs[:num_points // 2], np.abs(fft_result)[:num_points // 2])
    plt.title(f'FFT with {num_points} Points')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Different numbers of points for FFT
num_points_list = [sampling_rate, 2*sampling_rate, 4*sampling_rate]

# Plot FFT with different number of points
for num_points in num_points_list:
    plot_fft(signal, sampling_rate, num_points)
