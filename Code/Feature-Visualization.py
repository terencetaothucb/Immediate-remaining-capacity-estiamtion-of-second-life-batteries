import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from multiple files
file_path_list = ['data/data_Cylind21.xlsx', 'data/data_Pouch31.xlsx', 'data/data_Cylind21.xlsx']

# Set up the figure and axes for 1x3 subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 6))  # Adjust figsize for better visibility
cmap = plt.get_cmap('viridis', 10)  # Use a colormap with 100 colors

# Loop through each file in the file path list and corresponding axes
for ax, file_path in zip(axes, file_path_list):
    data = pd.read_excel(file_path, sheet_name="All")
    Fts = data.loc[:, 'U1':'U21'].values
    SOC = data['SOC'].values
    SOH = data['SOH'].values

    # Unique SOC values
    unique_SOCs = np.unique(SOC)
    norm = plt.Normalize(SOC.min(), SOC.max())

    # For each unique SOC value, scatter plot U1 vs SOH
    for u_soc in unique_SOCs:
        mask = SOC == u_soc
        sc = ax.scatter(SOH[mask], Fts[mask, 0], c=[u_soc]*np.sum(mask), cmap=cmap, norm=norm)

    # Extract the filename without the path and extension for the title
    file_name = file_path.split('/')[-1].replace('data_', '').replace('.xlsx', '')
    
    # Set labels and titles for each subplot
    ax.set_xlim([0.4, 1])
    ax.set_xlabel('SOH')
    ax.set_ylabel('Dimension of Voltage Dynamics U1 [V]')
    ax.set_title(f'{file_name}')  # Use the cleaned filename as title

# Create a single colorbar for all subplots
cbar = plt.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('SOC value')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leave space for the colorbar
plt.show()