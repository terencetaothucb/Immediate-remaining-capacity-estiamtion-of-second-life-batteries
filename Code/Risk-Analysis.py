import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel files
file_path_31 = 'Code/Cylind21toPouch31.xlsx'
file_path_52 = 'Code/Cylind21toPouch52.xlsx'

# Load the SOH Error Analysis sheet from both files
soh_error_31 = pd.read_excel(file_path_31, sheet_name="SOH Error Analysis")
soh_error_52 = pd.read_excel(file_path_52, sheet_name="SOH Error Analysis")

# Filter data for source and target domains
soh_error_31_source = soh_error_31[soh_error_31['Domain'] == 'Source']
soh_error_31_target = soh_error_31[soh_error_31['Domain'] == 'Target']

soh_error_52_source = soh_error_52[soh_error_52['Domain'] == 'Source']
soh_error_52_target = soh_error_52[soh_error_52['Domain'] == 'Target']

# Multiply SOH Error by 100 using .loc to avoid SettingWithCopyWarning
soh_error_31_source.loc[:, 'SOH Error'] = 100 * soh_error_31_source['SOH Error'] 
soh_error_31_target.loc[:, 'SOH Error'] = 100 * soh_error_31_target['SOH Error']
soh_error_52_source.loc[:, 'SOH Error'] = 100 * soh_error_52_source['SOH Error']
soh_error_52_target.loc[:, 'SOH Error'] = 100 * soh_error_52_target['SOH Error']

# Function to plot histograms and cumulative distribution with 95% marker and legends inside the upper right corner
def plot_soh_cdf_histograms_with_95_marker_legend(source_data, target_data, dataset_name):
    source_total = len(source_data)
    target_total = len(target_data)

    # Create a new figure for source domain
    plt.figure(figsize=(8, 6))
    
    # Source domain histogram
    plt.hist(source_data['SOH Error'], bins=int(0.26 / 0.01), alpha=0.5, label=f'{dataset_name} (Source)', color='orange', 
             weights=[1/source_total]*source_total, cumulative=False)
    plt.title(f'{dataset_name} Source Domain')
    plt.xlabel('RRC Error [%]')
    plt.ylabel('Frequency [-]')
    plt.ylim(0, 0.8)
    plt.xlim(0, 26)
    plt.grid(False)

    # CDF for source domain using np.histogram and plt.step to avoid the vertical line
    ax_twin_0 = plt.gca().twinx()
    hist, bin_edges = np.histogram(source_data['SOH Error'], bins=int(0.26 / 0.01), weights=[1/source_total]*source_total)
    cdf = np.cumsum(hist)
    ax_twin_0.step(bin_edges[:-1], cdf, where='post', color='orange', label='CDF (Source)', linewidth=2)
    ax_twin_0.set_ylim(0, 1.0)
    ax_twin_0.set_ylabel('Cumulative Frequency')

    # Find the SOH corresponding to 95% cumulative frequency for source
    soh_95_source_idx = (cdf >= 0.95).argmax()
    soh_95_source = bin_edges[soh_95_source_idx]
    ax_twin_0.axvline(soh_95_source, color='black', linestyle='--', label=f'95% at RRC={soh_95_source:.3f}')
    
    # Add legend inside the upper right corner of the plot
    plt.legend(loc='upper right')

    # Show the plot for source domain
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # save the figure with the name of the dataset and domain jpg 300dpi
    plt.savefig(f'{dataset_name}_source_domain.jpg', dpi=300)
    plt.show()

    # Create a new figure for target domain
    plt.figure(figsize=(8, 6))

    # Target domain histogram
    plt.hist(target_data['SOH Error'], bins=int(0.26 / 0.01), alpha=0.5, label=f'{dataset_name} (Target)', color='blue', 
             weights=[1/target_total]*target_total, cumulative=False)
    plt.title(f'{dataset_name} Target Domain')
    plt.xlabel('RRC Error [%]')
    plt.ylabel('Frequency [-]')
    plt.ylim(0, 0.8)
    plt.xlim(0, 26)
    plt.grid(False)

    # CDF for target domain using np.histogram and plt.step to avoid the vertical line
    ax_twin_1 = plt.gca().twinx()
    hist_target, bin_edges_target = np.histogram(target_data['SOH Error'], bins=int(0.26 / 0.01), weights=[1/target_total]*target_total)
    cdf_target = np.cumsum(hist_target)
    ax_twin_1.step(bin_edges_target[:-1], cdf_target, where='post', color='blue', label='CDF (Target)', linewidth=2)
    ax_twin_1.set_ylim(0, 1.0)
    ax_twin_1.set_ylabel('Cumulative Frequency')

    # Find the SOH corresponding to 95% cumulative frequency for target
    soh_95_target_idx = (cdf_target >= 0.95).argmax()
    soh_95_target = bin_edges_target[soh_95_target_idx]
    ax_twin_1.axvline(soh_95_target, color='black', linestyle='--', label=f'95% at RRC={soh_95_target:.3f}')
    
    # Add legend inside the upper right corner of the plot
    plt.legend(loc='upper right')

    # Show the plot for target domain
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # save the figure with the name of the dataset and domain jpg 300dpi
    plt.savefig(f'{dataset_name}_target_domain.jpg', dpi=300)
    plt.show()

# Plot for Source and Target Domain of Cylind21toPouch31
plot_soh_cdf_histograms_with_95_marker_legend(soh_error_31_source, soh_error_31_target, 'Cylind21toPouch31')

# Plot for Source and Target Domain of Cylind21toPouch52
plot_soh_cdf_histograms_with_95_marker_legend(soh_error_52_source, soh_error_52_target, 'Cylind21toPouch52')