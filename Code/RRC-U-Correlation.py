import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load data from multiple files
file_path_list = ['data/data_Cylind21.xlsx', 'data/data_Pouch31.xlsx', 'data/data_Cylind21.xlsx']

# Initialize the plot with 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
fig.subplots_adjust(right=0.85)

# Define the colormap
cmap = plt.get_cmap('viridis')

# Loop through each file and generate subplots
for i, file_path in enumerate(file_path_list):
    # Load data from Excel
    data = pd.read_excel(file_path, sheet_name="All")

    # Extract features and target variables
    Fts = data.loc[:, 'U1':'U21'].values
    SOC = data['SOC'].values
    SOH = data['SOH'].values

    # Initialize an empty DataFrame to store correlation results
    correlation_df = pd.DataFrame()

    # Group data by SOC and calculate the correlation for each group
    for soc_value, group_data in data.groupby('SOC'):
        correlations = []
        features = ['U' + str(j) for j in range(1, 22)]
        for feature in features:
            corr, _ = pearsonr(group_data[feature], group_data['SOH'])
            correlations.append(corr)
        correlation_df[soc_value] = correlations

    correlation_df.index = features

    # Generate X, Y coordinates
    X = np.array(correlation_df.columns)
    Y = np.arange(len(correlation_df.index))
    Z = correlation_df.values

    # Extract the filename without the path and extension for the title
    file_name = file_path.split('/')[-1].replace('data_', '').replace('.xlsx', '')

    # Create the contour plot in the first row (for SOC grouping)
    cp = axes[0, i].contourf(X, Y, Z, levels=100, cmap=cmap)
    axes[0, i].set_title(f'{file_name} (Grouped by SOC)')  # Use the extracted file name
    axes[0, i].set_xlabel('SOC[%]')
    axes[0, i].set_yticks(Y)
    axes[0, i].set_yticklabels(correlation_df.index)

    # Set x-ticks from 5 to 50 with intervals of 5
    axes[0, i].set_xticks(np.arange(5, 55, 5))
    axes[0, i].set_xticklabels(np.arange(5, 55, 5))

    if i == 0:
        axes[0, i].set_ylabel('Dimensions of Voltage Dynamics')

    # Now calculate the correlation without grouping by SOC (mixed SOC)
    mixed_correlations = []
    for feature in features:
        corr, _ = pearsonr(data[feature], data['SOH'])
        mixed_correlations.append(corr)

    # Normalize Pearson coefficients for colormap mapping
    norm = plt.Normalize(vmin=-1, vmax=1)
    colors = cmap(norm(mixed_correlations))

    # Create a scatter plot for the mixed SOC correlations in the second row
    scatter = axes[1, i].scatter(mixed_correlations, np.arange(len(features)), c=mixed_correlations, cmap=cmap, norm=norm, edgecolor='black')
    axes[1, i].set_title(f'{file_name} (Mixed SOC)')
    axes[1, i].set_xlabel('Pearson Correlation')
    axes[1, i].set_yticks(np.arange(len(features)))
    axes[1, i].set_yticklabels(features)

    # Set x-ticks for the second row to be only one digit
    axes[1, i].set_xticks(np.arange(-1, 1.1, 0.5))  # Adjust the range as necessary
    axes[1, i].set_xticklabels(np.round(np.arange(-1, 1.1, 0.5), 1))  # Round to one decimal place

    if i == 0:
        axes[1, i].set_ylabel('Dimensions of Voltage Dynamics')

# Create a single colorbar for all contour plots on the right side of the figure
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
cbar = fig.colorbar(cp, cax=cbar_ax, label='Pearson Correlation', orientation='vertical')

cbar.ax.tick_params(labelsize=10)  
cbar.formatter.set_powerlimits((0, 0))  
cbar.formatter.set_useOffset(False)  
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))  

plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, hspace=0.4)

plt.savefig('Code/Corr_Analysis.jpeg', dpi=300)
plt.show()