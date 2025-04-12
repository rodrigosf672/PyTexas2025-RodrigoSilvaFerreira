"""
Author: Rodrigo Silva Ferreira
Talk: Signal Processing in Electrochemistry with Python: Applications to the US Opioids Crisis
Date: 12APR2025 | Event: PyTexas 2025 | Location: Austin, TX
Purpose: Compare results between Script01 and Script02, to determine the impact of using HyperOpt on selecting parameters for SavGol Filter.
Expectation: Unsure... pure curiosity! Does using HyperOpt help achieve better results? If not, are differences statistically significant?
Reference for CV Data: Copley, G.; Gibson, E. Cyclic Voltammetry of a Cobaloxime Catalyst raw data. Newcastle University: 2019 (licensed under CC BY 4.0).
"""

# Import libraries
import pickle
import numpy as np
from scipy.stats import ttest_ind
import os

# Load results from Script01 and Script02
folder_path = 'Script03-PickleFiles'
file_path_script01 = os.path.join(folder_path, 'results_script01.pkl')
with open(file_path_script01, 'rb') as f:
    results_script01 = pickle.load(f)
file_path_script02 = os.path.join(folder_path, 'results_script02.pkl')
with open(file_path_script02, 'rb') as f:
    results_script02 = pickle.load(f)

# Calculate average of the average MSLEs and standard deviations for Script01 and Script02
avg_msle_script01 = np.mean(results_script01['avg_msle'])
std_dev_msle_script01 = np.mean(results_script01['std_dev_msle'])
avg_msle_script02 = np.mean(results_script02['avg_msle'])
std_dev_msle_script02 = np.mean(results_script02['std_dev_msle'])
print(f"Script01 - Average MSLE: {avg_msle_script01:.4f}, Standard Deviation: {std_dev_msle_script01:.4f}")
print(f"Script02 - Average MSLE: {avg_msle_script02:.4f}, Standard Deviation: {std_dev_msle_script02:.4f}")

# Calculate average of the average MSEs and standard deviations for Script01 and Script02
avg_mse_script01 = np.mean(results_script01['avg_mse'])
std_dev_mse_script01 = np.mean(results_script01['std_dev_mse'])
avg_mse_script02 = np.mean(results_script02['avg_mse'])
std_dev_mse_script02 = np.mean(results_script02['std_dev_mse'])
print(f"Script01 - Average MSE: {avg_mse_script01:.4f}, Standard Deviation: {std_dev_mse_script01:.4f}")
print(f"Script02 - Average MSE: {avg_mse_script02:.4f}, Standard Deviation: {std_dev_mse_script02:.4f}")

# Perform two-sample t-test between averages for MSLE
t_stat_msle, p_value_msle = ttest_ind(results_script01['avg_msle'], results_script02['avg_msle'])
print(f"T-statistic (MSLE): {t_stat_msle:.4f}, P-value: {p_value_msle:.4f}")

# Determine if the difference is statistically significant for MSLE
alpha = 0.05
if p_value_msle < alpha:
    print("There is a statistically significant difference between the two sets of average MSLEs.")
else:
    print("There is NO statistically significant difference between the two sets of average MSLEs.")

# Perform two-sample t-test between averages for MSE
t_stat_mse, p_value_mse = ttest_ind(results_script01['avg_mse'], results_script02['avg_mse'])
print(f"T-statistic (MSE): {t_stat_mse:.4f}, P-value: {p_value_mse:.4f}")

# Determine if the difference is statistically significant for MSE
if p_value_mse < alpha:
    print("There is a statistically significant difference between the two sets of average MSEs.")
else:
    print("There is NO statistically significant difference between the two sets of average MSEs.")
