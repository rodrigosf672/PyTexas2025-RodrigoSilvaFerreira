"""
Author: Rodrigo Silva Ferreira
Talk: Signal Processing in Electrochemistry with Python: Applications to the US Opioids Crisis
Date: 28JUL2024 | Event: PyOhio 2024 | Location: Cleveland, OH
Purpose: Demonstrate how the Savitzky-Golay Filter can be helpful to optimize CV signals, with no pre-set parameters.
Method: Use hyperopt to find the best parameters, then plot CV data.
Expectation: Four plots containing original CV data, noisy data, and filtered data with the best combinations of window sizes and polynomial orders.
Reference for CV Data: Copley, G.; Gibson, E. Cyclic Voltammetry of a Cobaloxime Catalyst raw data. Newcastle University: 2019 (licensed under CC BY 4.0).
"""

# Import the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import f_oneway
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle

# Load data from CSV file and extract potentials and currents from columns. Set folder for output plots.
file_path = 'CyclicVoltammetryData_RedoxReaction.csv'
data = pd.read_csv(file_path)
E = data['E /V']
I = data['I /uA']
output_folder = 'Script02-Plots'
os.makedirs(output_folder, exist_ok=True)

# Define varying levels of noise to the original data. Set seed for random noise to ensure reproducibility.
noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
np.random.seed(30)

# Define the search space for window sizes and polynomial orders.
def search_space():
    window_length = hp.choice('window_length', [i for i in range(5, 102, 2)])  # Odd numbers between 5 and 101
    poly_order = hp.choice('poly_order', [i for i in range(1, 8, 2)])  # Odd numbers between 1 and 7
    return {'window_length': window_length, 'poly_order': poly_order}

# Define objective function to search for parameters that would lead to the lowest MSLE/MSE values.
def objective(params):
    window_length = params['window_length']
    poly_order = params['poly_order']
    if window_length <= poly_order:
        return {'loss': float('inf'), 'status': STATUS_OK}
    msle_scores = []
    mse_scores = []
    for noise in noise_levels:
        I_noisy = I + np.random.normal(0, max(I) * noise, size=I.shape)
        I_restored = savgol_filter(I_noisy, window_length, poly_order)
        valid_indices = (E >= -0.5) & (E <= 0.2)
        I_original_valid = I[valid_indices]
        I_restored_valid = I_restored[valid_indices]
        valid_mask_msle = (I_original_valid > 0) & (I_restored_valid > 0)
        I_original_valid_msle = I_original_valid[valid_mask_msle]
        I_restored_valid_msle = I_restored_valid[valid_mask_msle]
        msle = np.mean((np.log(I_original_valid_msle + 1) - np.log(I_restored_valid_msle + 1)) ** 2)
        msle_scores.append(msle)
        mse = np.mean((I_original_valid - I_restored_valid) ** 2)
        mse_scores.append(mse)  
    avg_msle = np.mean(msle_scores)
    avg_mse = np.mean(mse_scores)
    return {'loss': avg_msle, 'status': STATUS_OK, 'msle': avg_msle, 'mse': avg_mse}

# Perform hyperparameter optimization using hyperopt. Let's only use 20 attempts.
trials = Trials()
best_params = fmin(fn=objective, space=search_space(), algo=tpe.suggest, max_evals=20, trials=trials)

# Extract the top four parameter combinations ensuring valid Window Lengths and Polynomial Orders.
valid_combinations = []
for i in range(len(trials.trials)):
    window_length = trials.trials[i]['misc']['vals']['window_length'][0] * 2 + 5
    poly_order = trials.trials[i]['misc']['vals']['poly_order'][0] * 2 + 1
    if window_length > poly_order:
        valid_combinations.append((window_length, poly_order))
    if len(valid_combinations) == 4:
        break
all_msle_results = {combo: [] for combo in valid_combinations}
all_mse_results = {combo: [] for combo in valid_combinations}

# Add varying levels of random noise to the original data.
for noise in noise_levels:
    msle_results = []  # Reset for each noise level
    mse_results = [] # Reset for each noise level
    I_noisy = I + np.random.normal(0, max(I) * noise, size=I.shape)  # Calculate max_current within the loop
    # Prepare plots 2x2 with the respective sizes.
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    # Apply the Savitzky-Golay filter and plot them.
    for i, (window_length, poly_order) in enumerate(valid_combinations):
        I_restored = savgol_filter(I_noisy, window_length, poly_order)
        
        # Define data within specified potential range of the observed peak (-0.5 to 0.2V)
        valid_indices = (E >= -0.5) & (E <= 0.2)
        E_valid = E[valid_indices]
        I_original_valid = I[valid_indices]
        I_restored_valid = I_restored[valid_indices]
        
        # Filter out non-positive values to avoid runtime warnings
        valid_mask = (I_original_valid > 0) & (I_restored_valid > 0)
        I_original_valid = I_original_valid[valid_mask]
        I_restored_valid = I_restored_valid[valid_mask]
        
        # Calculate mean squared logarithmic error (MSLE) of the results
        msle = np.mean((np.log(I_original_valid + 1) - np.log(I_restored_valid + 1)) ** 2)
        msle_results.append((window_length, poly_order, msle))
        all_msle_results[(window_length, poly_order)].append(msle)

        # Calculate mean squared error (MSE) of the results
        mse = np.mean((I_original_valid - I_restored_valid) ** 2)
        mse_results.append((window_length, poly_order, mse))
        all_mse_results[(window_length, poly_order)].append(mse)
        
        # Define parameters for plots
        axs[i].plot(E, I, label='Original CV Data', color='blue', linewidth=2)
        axs[i].plot(E, I_noisy, label='Noisy CV Data', color='red', linewidth=1, alpha=0.5)
        axs[i].plot(E, I_restored, label='Filtered CV Data', color='green', linewidth=2)
        axs[i].set_title(f"Window Size: {window_length}, Polynomial Order: {poly_order}, Noise: {noise}, MSLE: {msle:.2f}, MLE: {mse:.2f}")
        axs[i].set_xlabel('Potential (V)')
        axs[i].set_ylabel('Current (µA)')
        axs[i].legend()
        axs[i].invert_xaxis()
        axs[i].set_xticks(np.arange(E.max(), E.min() - 0.1, -0.1))
    
    # Print results in the terminal
    print(f"Results for Noise Level: {noise}")
    for (window_length, poly_order, msle), (_, _, mse) in zip(msle_results, mse_results):
        print(f"Window Size: {window_length}, Polynomial Order: {poly_order}, MSLE: {msle:.2f}, MSE: {mse:.2f}")
    
    # Create plots and save them in Script02-Plots
    filename = os.path.join(output_folder, f'CV_Signal_Noise{noise}_WithHyperOpt.png')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Calculate average MSLE/MSE and standard deviation for each model
avg_msle = [np.mean(all_msle_results[combo]) for combo in valid_combinations]
std_dev_msle = [np.std(all_msle_results[combo]) for combo in valid_combinations]
avg_mse = [np.mean(all_mse_results[combo]) for combo in valid_combinations]
std_dev_mse = [np.std(all_mse_results[combo]) for combo in valid_combinations]

# Perform ANOVA test
anova_result_msle = f_oneway(*[all_msle_results[combo] for combo in valid_combinations])
p_value_msle = anova_result_msle.pvalue
anova_result_mse = f_oneway(*[all_mse_results[combo] for combo in valid_combinations])
p_value_mse = anova_result_mse.pvalue

# Plot bar charts of average MSLE/MSE values for each model with error bars
models = [f'{combo[0]}, {combo[1]}' for combo in valid_combinations]
positions = np.arange(len(models))
width = 0.4

# Plot for MSLE
fig, ax = plt.subplots()
bars = ax.bar(positions, avg_msle, width, yerr=std_dev_msle, capsize=5, color='skyblue', edgecolor='black')
ax.set_xlabel('Model (Window Size, Polynomial Order)')
ax.set_ylabel('Average MSLE')
ax.set_xticks(positions)
ax.set_xticklabels(models)
ax.set_title(f'Resistance to Noise - MSLE with Errors\nOne-Way ANOVA p-value: {p_value_msle:.4f}')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'Resistance_to_Noise_MSLE_WithHyperOpt.jpeg'), dpi=300)

# Plot for MSE
fig, ax = plt.subplots()
bars = ax.bar(positions, avg_mse, width, yerr=std_dev_mse, capsize=5, color='lightgreen', edgecolor='black')
ax.set_xlabel('Model (Window Size, Polynomial Order)')
ax.set_ylabel('Average MSE')
ax.set_xticks(positions)
ax.set_xticklabels(models)
ax.set_title(f'Resistance to Noise - MSE with Errors\nOne-Way ANOVA p-value: {p_value_mse:.4f}')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'Resistance_to_Noise_MSE_WithHyperOpt.jpeg'), dpi=300)

# Store results in a Pickle file for statistical analysis later
pickle_folder = 'Script03-PickleFiles'
os.makedirs(pickle_folder, exist_ok=True)
pickle_file = os.path.join(pickle_folder, 'results_script02.pkl')
results_script02 = {
    'avg_msle': avg_msle,
    'std_dev_msle': std_dev_msle,
    'avg_mse': avg_mse,
    'std_dev_mse': std_dev_mse
}
with open(pickle_file, 'wb') as f:
    pickle.dump(results_script02, f)