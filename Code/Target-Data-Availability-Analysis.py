import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data for MAPE (Target and Source domain separated)
data_mape = {
    'Target Size': ['1/3 = 700', '1/5 = 420', '1/10 = 210', '1/20 = 105', '1/30 = 70', '1/40 = 52', '1/50 = 42'],
    'Pouch31 (MAPE % - Target)': [1.7, 1.8, 1.6, 2.8, 3.7, 3.7, 3.6],
    'Pouch52 (MAPE % - Target)': [3.4, 3.2, 3.0, 5.9, 6.0, 8.1, 7.2],
    'Pouch31 (MAPE % - Source)': [3.2, 3.4, 3.7, 3.9, 4.2, 4.0, 3.8],
    'Pouch52 (MAPE % - Source)': [3.1, 3.6, 3.3, 3.5, 3.4, 3.4, 3.6]
}

# Data for ρ (Target and Source domain separated)
data_rho = {
    'Target Size': ['1/3 = 700', '1/5 = 420', '1/10 = 210', '1/20 = 105', '1/30 = 70', '1/40 = 52', '1/50 = 42'],
    'Pouch31 (rho - Target)': [0.89, 0.90, 0.90, 0.85, 0.75, 0.70, 0.73],
    'Pouch52 (rho - Target)': [0.93, 0.96, 0.95, 0.89, 0.88, 0.82, 0.84],
    'Pouch31 (rho - Source)': [0.86, 0.84, 0.82, 0.84, 0.82, 0.82, 0.83],
    'Pouch52 (rho - Source)': [0.88, 0.85, 0.86, 0.86, 0.85, 0.86, 0.84]
}

# Create DataFrames
df_mape = pd.DataFrame(data_mape)
df_rho = pd.DataFrame(data_rho)

# Set index to Target Size for heatmap compatibility
df_mape.set_index('Target Size', inplace=True)
df_rho.set_index('Target Size', inplace=True)

# Plotting heatmaps with target, target, source, source columns using light colormap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# MAPE Heatmap (Using a lighter colormap like YlGnBu)
sns.heatmap(df_mape, annot=True, fmt='.1f', cmap='YlGnBu', cbar=True, ax=ax1)
ax1.set_title('MAPE (%) Heatmap (Target, Target, Source, Source)', fontsize=14)

# ρ Heatmap (Using a "coolwarm" colormap)
sns.heatmap(df_rho, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, ax=ax2)
ax2.set_title('rho Heatmap (Target, Target, Source, Source)', fontsize=14)

# Show plots
plt.tight_layout()
# save the plot as a file jpg 300 dpi
#plt.savefig('tgt-size-sensitivity.jpg', dpi=300)
plt.show()
