import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assume you have three dataset file paths
file_paths = ['data/data_Cylind21.xlsx', 'data/data_Pouch31.xlsx', 'data/data_Pouch52.xlsx']
domains = ['Cylind21', 'Pouch31', 'Pouch52']  # Corresponding domain labels

# Create an empty list to store datasets
all_data = []

# Iterate through all file paths, read data, and add domain labels
for file_path, domain in zip(file_paths, domains):
    data = pd.read_excel(file_path, sheet_name="All")
    
    # Extract Fts, SOC, and SOH data
    Fts = data.loc[:, 'U1':'U21']
    SOC = data['SOC']
    SOH = data['SOH']
    
    # Merge SOC, SOH, Fts, and add domain labels
    data_with_soc_soh = pd.concat([Fts, SOC, SOH], axis=1)
    data_with_soc_soh['domain'] = domain  # Add domain label
    all_data.append(data_with_soc_soh)

# Merge all datasets into a single DataFrame
final_data = pd.concat(all_data, ignore_index=True)

# Convert to long format for Seaborn visualization
final_data_long = pd.melt(final_data, id_vars=["domain", "SOC", "SOH"], 
                          value_vars=[f'U{i}' for i in range(1, 22)], 
                          var_name="Feature", value_name='Voltage dynamics [V]')

# Set global font size to 12
plt.rc('font', size=12)

# Create Figure 1: Set domain as hue
plt.figure(figsize=(10, 6))
sns.boxplot(x='Feature', y='Voltage dynamics [V]', hue='domain', data=final_data_long, palette="coolwarm", dodge=True)
plt.xticks(ticks=range(21), labels=[f'U{i}' for i in range(1, 22)], rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=10)
plt.tight_layout()
plt.title('Voltage dynamics with hue="domain"')

# Save the plot as a JPG file with 300 dpi
plt.savefig('Voltage dynamics with hue="domain".jpg', dpi=300)
plt.show()
