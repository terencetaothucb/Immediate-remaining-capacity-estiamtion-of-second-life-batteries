import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load data
file_path1 = 'data/data_Cylind21.xlsx'
file_path2 = 'data/data_Pouch31.xlsx'
file_path3 = 'data/data_Pouch52.xlsx'

data1 = pd.read_excel(file_path1, sheet_name="All")
Fts1 = data1.loc[:, 'U1':'U21'].values
SOC1 = data1['SOC'].values
SOH_values1 = np.unique(data1['SOH'].values)

data2 = pd.read_excel(file_path2, sheet_name="All")
Fts2 = data2.loc[:, 'U1':'U21'].values
SOC2 = data2['SOC'].values
SOH_values2 = np.unique(data2['SOH'].values)

data3 = pd.read_excel(file_path3, sheet_name="All")
Fts3 = data3.loc[:, 'U1':'U21'].values
SOC3 = data3['SOC'].values
SOH_values3 = np.unique(data3['SOH'].values)

# Create separate plot for SOH distribution
plt.figure(figsize=(8, 6))
sns.kdeplot(SOH_values1, label='Pouch52', fill=True, color='r', bw_adjust=1, alpha=0.3)
sns.kdeplot(SOH_values2, label='Pouch31', fill=True, color='g', bw_adjust=1, alpha=0.3)
sns.kdeplot(SOH_values3, label='Cylind21', fill=True, color='b', bw_adjust=1, alpha=0.3)
plt.xlabel('SOH')
plt.ylabel('Density')
plt.title('SOH Distribution')
plt.legend()


# Create figure
fig = plt.figure(figsize=(8, 8), facecolor='none')  # Setting figure background to transparent
ax = fig.add_subplot(111, projection='3d')

# Set 3D plot background color to transparent
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove gridlines
ax.grid(True)

# A parameter to increase visual distance between feature slices
distance_factor = 10000

# Loop over all the features to create separate KDE plots along the y-axis
for i in range(Fts1.shape[1]):
    feature_idx = i * distance_factor 
    
    # KDE computation for each dataset
    kde1 = gaussian_kde(Fts1[:, i])
    kde2 = gaussian_kde(Fts2[:, i])
    kde3 = gaussian_kde(Fts3[:, i])
    
    # Create an x range for evaluation of the KDE
    x_range = np.linspace(min(Fts1[:, i].min(), Fts2[:, i].min(), Fts3[:, i].min()),
                          max(Fts1[:, i].max(), Fts2[:, i].max(), Fts3[:, i].max()), 100)
    
    # Evaluate the densities
    density1 = kde1(x_range)
    density2 = kde2(x_range)
    density3 = kde3(x_range)
    
    # 3D plot: x = feature values, y = feature index, z = density
    ax.plot(x_range, np.full_like(x_range, feature_idx), density1, color='r')
    ax.plot(x_range, np.full_like(x_range, feature_idx), density2, color='g')
    ax.plot(x_range, np.full_like(x_range, feature_idx), density3, color='b')

    # Shaded region under plot
    for density, color in zip([density1, density2, density3], ['r', 'g', 'b']):
        verts = [list(zip(x_range, np.full_like(x_range, feature_idx), density))]
        poly = Poly3DCollection(verts, facecolors=color, linewidths=0.5, edgecolors=color, alpha=0.2)
        ax.add_collection3d(poly)

# Labeling and legend
ax.set_xlabel('Voltage')
ax.set_ylabel('Feature')
ax.set_zlabel('Density')

# Manually add legend
legend_elements = [plt.Line2D([0], [0], color='r', label='Pouch52'),
                   plt.Line2D([0], [0], color='g', label='Pouch31'),
                   plt.Line2D([0], [0], color='b', label='Cylind21')]
ax.legend(handles=legend_elements)

# Set y-ticks and their labels to represent features
ax.set_yticks([i * distance_factor for i in range(Fts1.shape[1])])
ax.set_yticklabels([f'U{i+1}' for i in range(Fts1.shape[1])])

plt.tight_layout()
plt.show()