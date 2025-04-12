"""
Author: Rodrigo Silva Ferreira
Talk: Signal Processing in Electrochemistry with Python: Applications to the US Opioids Crisis
Date: 12APR2025 | Event: PyTexas 2025 | Location: Austin, TX
Purpose: Demonstrate how Fourier Transform can be helpful to analyze CV signals by identifying distinctive chemical signatures through the harmonics.
Method: Plot CV data of Original, Simulated Interfering Substance, and Combined Signal. Then, perform Fourier Transform, highlighting different harmonics.
Expectation: CV plots for Original, Interfering Substance, and Combined Signal, along with the respective Fourier Transform with highlighted harmonics.
Reference for CV Data: Copley, G.; Gibson, E. Cyclic Voltammetry of a Cobaloxime Catalyst raw data. Newcastle University: 2019 (licensed under CC BY 4.0).
"""

# Import the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Load data from CSV file and extract potentials and currents from columns. Set folder for output plots.
file_path = 'CyclicVoltammetryData_RedoxReaction.csv'
data = pd.read_csv(file_path)
E = data['E /V'].to_numpy()
I = data['I /uA'].to_numpy()
output_folder = 'Script04-Plots'
os.makedirs(output_folder, exist_ok=True)

# Define N as the number of data points
N = len(I)

# Generate an overlapping signal by adding a sinusoidal component in order to simulate an interfering substance.
freq = 5  # Frequency of the sinusoidal signal (Hz)
amplitude = max(I) * 0.1  # Amplitude of the sinusoidal signal
I_interferingsubstance = I + amplitude * np.sin(2 * np.pi * freq * np.arange(N) / N)

# Create the combined signal by adding the Original and Interfering Substance signals
I_combined = (I + I_interferingsubstance)/2

# Perform Fourier Transform on the Original, Interfering Substance, and Combined signals
T = (E.max() - E.min()) / N  # Sample spacing
I_fft = fft(I)
I_freq = fftfreq(N, T)[:N//2]  # Positive frequencies
I_amplitude = 2.0/N * np.abs(I_fft[:N//2])  # Amplitude of the FFT
I_interferingsubstance_fft = fft(I_interferingsubstance)
I_interferingsubstance_amplitude = 2.0/N * np.abs(I_interferingsubstance_fft[:N//2])  # Amplitude of the FFT
I_combined_fft = fft(I_combined)
I_combined_amplitude = 2.0/N * np.abs(I_combined_fft[:N//2])  # Amplitude of the FFT

# Calculate the fundamental frequency and harmonics for the Original signal
fundamental_freq = I_freq[np.argmax(I_amplitude)]
harmonics = fundamental_freq * np.arange(1, 10)  # Up to the 9th harmonic

# Plot Original, Interfering Substance, and Combined CV data
plt.figure(figsize=(16, 8)) 
plt.plot(E, I, label='Original CV Data', color='green', linewidth=2, linestyle='--')
plt.plot(E, I_interferingsubstance, label='Interfering Substance CV Data', color='red', linewidth=2, linestyle='--')
plt.plot(E, I_combined, label='Combined CV Data', color='blue', linewidth=2, linestyle='-')
plt.title('Original, Interfering Substance, and Combined Cyclic Voltammetry Data')
plt.xlabel('Potential (V)')
plt.ylabel('Current (µA)')
plt.gca().invert_xaxis()
plt.xticks(np.arange(E.max(), E.min() - 0.1, -0.1))
plt.legend()
plt.tight_layout()

# Save CV data plot
filename_original = os.path.join(output_folder, 'Original_InterferingSubstance_Combined_CV.png')
plt.savefig(filename_original)
plt.close()

# Plot Fourier Transform of the original, Interfering Substance, and combined CV data
plt.figure(figsize=(16, 8))  
plt.plot(I_freq, I_amplitude, label='Original Fourier Transform', color='green', linewidth=2, linestyle='--')
plt.plot(I_freq, I_interferingsubstance_amplitude, label='Interfering Substance Fourier Transform', color='red', linewidth=2, linestyle='--')
plt.plot(I_freq, I_combined_amplitude, label='Combined Fourier Transform', color='blue', linewidth=2, linestyle='-')
plt.xlim(0, 15)  # Limit x-axis to range 0-15 Hz
plt.title('Fourier Transform of Original, Interfering Substance, and Combined CV Data (0-15 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# Highlight different harmonics
for harmonic in harmonics:
    if harmonic <= 15:  # Only plot harmonics within the x-axis range
        plt.axvline(harmonic, color='gray', linewidth=1, linestyle='--')
        plt.text(harmonic, plt.gca().get_ylim()[1] * 0.9, f'{harmonic:.2f} Hz', color='gray', ha='center')
plt.legend()
plt.tight_layout()

# Save the Fourier Transform Plot
filename_fourier = os.path.join(output_folder, 'FourierTransform_CV.png')
plt.savefig(filename_fourier)
plt.close()
