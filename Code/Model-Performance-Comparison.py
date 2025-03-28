import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create the updated dataset
data = {
    'Model': ['Linear Regression', 'Ridge Regression', 'Gaussian Process', 'SVM', 'k-NN', 'Random Forest', 'Deep neural network', 'Deep generative transfer\n(Our work)'],
    'Pouch31_MAPE_1': [100.72, 7.13, 62.21, 6.64, 8.23, 9.00, 10.51, 3.6],  # 1/50
    'Pouch52_MAPE_1': [62.23, 15.95, 37.78, 17.94, 17.03, 16.49, 10.31, 7.2],
    'Pouch31_rho_1': [0.19, 0.17, 0.002, 0.11, 0.05, -0.02, 0.07, 0.73],
    'Pouch52_rho_1': [0.01, 0.72, -0.104, 0.55, 0.34, 0.33, 0.69, 0.84],
    
    'Pouch31_MAPE_2': [100.5, 6.69, 25.98, 6.31, 7.53, 7.64, 9.33, 3.7],  # 1/40
    'Pouch52_MAPE_2': [66.35, 15.85, 19.03, 17.98, 17.45, 17.22, 11.54, 8.1],
    'Pouch31_rho_2': [0.19, 0.14, 0.181, 0.10, 0.10, 0.10, 0.34, 0.70],
    'Pouch52_rho_2': [-0.04, 0.71, 0.594, 0.55, 0.28, 0.26, 0.70, 0.82],
    
    'Pouch31_MAPE_3': [63.66, 6.99, 18.91, 6.03, 7.74, 7.43, 6.22, 3.7],  # 1/30
    'Pouch52_MAPE_3': [67.76, 17.47, 24.21, 18.08, 17.50, 17.20, 11.82, 6.0],
    'Pouch31_rho_3': [0.08, 0.14, 0.154, 0.17, 0.05, 0.15, 0.51, 0.75],
    'Pouch52_rho_3': [-0.03, 0.63, 0.424, 0.57, 0.21, 0.24, 0.69, 0.88],
    
    'Pouch31_MAPE_4': [48.46, 7.61, 28.46, 6.10, 7.71, 8.00, 5.15, 2.8],  # 1/20
    'Pouch52_MAPE_4': [86.03, 16.96, 22.92, 17.64, 17.14, 16.26, 9.64, 5.9],
    'Pouch31_rho_4': [0.13, 0.17, 0.096, 0.19, 0.01, 0.08, 0.64, 0.85],
    'Pouch52_rho_4': [0.02, 0.65, 0.392, 0.67, 0.25, 0.34, 0.74, 0.89],
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Update domain_labels
domain_labels = ['1/50 (42)', '1/40 (52)', '1/30 (70)', '1/20 (105)']

# Assign different markers to different models
markers = ['o', 's', 'D', '^', 'v', 'x', 'p', '+']  # Removed the pentagram marker for DNN, changed to 'p', leaving the pentagram for Our work

# Set global font size
plt.rcParams.update({'font.size': 14})  # Set global font size to 12

# Generate a line chart and highlight "Our work" (pentagram marker, red line)
def plot_2by2_line_chart_log_scale():
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex=False)  # Adjust overall size to ensure subplots are square
    x = np.arange(len(domain_labels))

    metrics = ['Pouch31_MAPE', 'Pouch52_MAPE', 'Pouch31_rho', 'Pouch52_rho']

    # Plot line charts for each model
    for idx, metric in enumerate(metrics):
        row, col = divmod(idx, 2)
        ax = axs[row, col]

        for i, model in enumerate(df['Model']):
            y_values = df[df['Model'] == model][[f'{metric}_1', f'{metric}_2', f'{metric}_3', f'{metric}_4']].values.flatten()

            if model == 'Deep generative transfer\n(Our work)':
                linestyle = '-'
                marker = '*'
                color = 'red'
                ax.plot(x, y_values, linestyle=linestyle, marker=marker, label=model, linewidth=2, color=color)
            else:
                linestyle = '--'
                marker = markers[i]
                ax.plot(x, y_values, linestyle=linestyle, marker=marker, label=model, linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(domain_labels, rotation=0)

        # Use log scale for MAPE
        if 'MAPE' in metric:
            ax.set_yscale('log')  # Use logarithmic scale
            ax.set_ylim(1, 150)  # Adjust log scale range
            ax.set_ylabel('Mean absolute percentage error [%]')
        else:
            ax.set_ylim(-0.2, 1)
            ax.set_ylabel('Pearson correlation [-]')

        ax.set_xlabel('Field data availability [-]')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.78, 0.5), ncol=1, frameon=False, fontsize=12)  # Legend at the middle right, font size 12

    # Adjust spacing between subplots
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)  # Set appropriate spacing to avoid overlap
    plt.savefig('benchmarking.jpg', dpi=300)
    plt.show()

# Call function to generate 2x2 layout line chart with log scale
plot_2by2_line_chart_log_scale()
