"""
Author: Rodrigo Silva Ferreira
Talk: Signal Processing in Electrochemistry with Python: Applications to the US Opioids Crisis
Date: 12APR2025 | Event: PyTexas 2025 | Location: Austin, TX
Purpose: Demonstrate how Wavelet Transform can be helpful to analyze noisy CV signals and distinguish between different signals.
Method: Plot CV data with noise and perform Wavelet Transform to identify key features.
Expectation: Plots containing original CV data with noise and its Wavelet Transform.
Reference for CV Data: Copley, G.; Gibson, E. Cyclic Voltammetry of a Cobaloxime Catalyst raw data. Newcastle University: 2019 (licensed under CC BY 4.0).
"""

# Import the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Load data from CSV file and extract potentials and currents from columns. Set folder for output plots.
file_path = 'CyclicVoltammetryData_RedoxReaction.csv'
data = pd.read_csv(file_path)
E = data['E /V'].to_numpy()
I = data['I /uA'].to_numpy()
output_folder = 'Script05-Plots'
os.makedirs(output_folder, exist_ok=True)

# Define N as the number of data points
N = len(I)

# Define noise levels
noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Set seed for random noise to ensure reproducibility.
np.random.seed(30)

# Perform analysis for each noise level
for noise_level in noise_levels:
    # Generate a noisy signal by adding random noise to the original data
    noise_amplitude = noise_level * max(I)  # Amplitude of the noise
    I_noisy = I + noise_amplitude * np.random.normal(size=N)
    # Perform Wavelet Transform on both the original and noisy signals
    wavelet = 'db4'
    coeffs_original = pywt.wavedec(I, wavelet, level=5)
    coeffs_noisy = pywt.wavedec(I_noisy, wavelet, level=5)
    # Plot original and noisy CV data
    plt.figure(figsize=(16, 8))
    plt.plot(E, I, label='Original CV Data', color='green', linewidth=2)
    plt.plot(E, I_noisy, label=f'Noisy CV Data', color='red', linewidth=2, linestyle='--')
    plt.title(f'Original and Noisy Cyclic Voltammetry Data (Noise Level {noise_level})')
    plt.xlabel('Potential (V)')
    plt.ylabel('Current (µA)')
    plt.gca().invert_xaxis()
    plt.xticks(np.arange(E.max(), E.min() - 0.1, -0.1))
    plt.legend()
    plt.tight_layout()
    # Save the first plot
    filename_original = os.path.join(output_folder, f'Original_Noisy_CV_{noise_level}.png')
    plt.savefig(filename_original)
    plt.close()
    # Plot Wavelet Transform coefficients for both the original and noisy signals
    plt.figure(figsize=(16, 16)) 
    for i, (coeff_orig, coeff_noisy) in enumerate(zip(coeffs_original, coeffs_noisy)):
        plt.subplot(len(coeffs_original), 1, i + 1)
        plt.plot(coeff_orig, label=f'Level {i+1} Coefficients (Original)', color='green')
        plt.plot(coeff_noisy, label=f'Level {i+1} Coefficients (Noisy)', color='red', linestyle='--')
        plt.legend()
        plt.xlabel('Position in Time-Frequency Domain')
        plt.ylabel('Coefficient Magnitude')
    plt.suptitle(f'Wavelet Transform Coefficients of Original and Noisy CV Data (Noise Level {noise_level})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the second plot
    filename_wavelet = os.path.join(output_folder, f'Wavelet_Transform_Coefficients_{noise_level}.png')
    plt.savefig(filename_wavelet)
    plt.close()
