import os
import random
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import mse
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Set seeds
seed_value = 42  # You can choose any integer here
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Limit threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# Load data
file_path = 'data/data_Cylind21.xlsx'
#file_path = 'data/data_Pouch31.xlsx'
#file_path = 'data/data_Pouch52.xlsx'
sampling_multiplier = 10  # Hyperparameter to control sampling size

data = pd.read_excel(file_path, sheet_name="All")
Fts = data.loc[:, 'U1':'U21'].values
SOC = data['SOC'].values
SOE = data['SOE'].values
SOH_values = np.unique(data['SOH'].values)

# Variables for VAE
original_dim = Fts.shape[1] + 1  # Number of features + SOC
intermediate_dim = 128
latent_dim = 2
batch_size = 32
epochs = 1000

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1, seed=0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAE Architecture - Encoder
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(x, z)

# VAE Architecture - Decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')  # Use sigmoid because of MinMax scaling
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
decoder = Model(decoder_input, _x_decoded_mean)

vae_output = decoder(encoder(x))
vae = Model(x, vae_output)

xent_loss = original_dim * mse(x, vae_output)
kl_weight = 1
kl_loss = - kl_weight * 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam())

augmented_data = []
augmented_SOE_list = []

for soh in SOH_values:
    mask = data['SOH'] == soh
    current_Fts = Fts[mask]
    current_SOC = SOC[mask][:, np.newaxis]
    combined_data = np.hstack([current_SOC, current_Fts])
    scaler = MinMaxScaler().fit(combined_data)
    combined_data_normalized = scaler.transform(combined_data)
    vae.fit(combined_data_normalized, epochs=epochs, batch_size=batch_size)

    num_samples = len(combined_data_normalized) * sampling_multiplier
    random_latent_values = K.random_normal(shape=(num_samples, latent_dim),seed=0)
    new_data_samples_normalized = decoder.predict(random_latent_values)
    new_data_samples = scaler.inverse_transform(new_data_samples_normalized)
    augmented_data.append(new_data_samples)

    augmented_SOC_current = new_data_samples[:, 0]
    augmented_SOE_current = augmented_SOC_current * soh   
    augmented_SOE_list.append(augmented_SOE_current)

# Encode the input data to obtain latent space values
encoded_data = encoder.predict(combined_data_normalized)
all_augmented_data = np.vstack(augmented_data)
augmented_SOE = np.concatenate(augmented_SOE_list)
augmented_SOC = all_augmented_data[:, 0]
augmented_Fts = all_augmented_data[:, 1:]

# Plotting the original vs augmented data using SOC
# Adjusting plot colors and settings for visual appeal
size = 10
plt.figure(figsize=(4, 4))
# Scatter plot with better color and transparency for visibility
plt.scatter(SOC, Fts[:, 0], c='#c5e7e8', label='Tested Data', s=50, alpha=0.7, edgecolors='#29b4b6')  # Light blue with black edge
plt.scatter(augmented_SOC, augmented_Fts[:, 0], c='#fbd2cb', label='Generated Data', s=50, alpha=0.1,edgecolors='#f0776d')  # Orange with transparency and black edge
# Title and labels with size adjustment
plt.xlabel("SOC [%]", fontsize=size)
plt.ylabel('Dimension of Voltage Dynamics U1 [V]', fontsize=size)
# Axis ticks and legend with size adjustments
plt.xticks(np.arange(5, 55, 5), fontsize=size)
plt.yticks(fontsize=size)
# y limit from 3.4 to 4.2
plt.ylim(3.4, 4.2)
plt.legend(fontsize=size)
# Layout adjustment for better spacing
plt.tight_layout()
# save the plot for file_path data to jpg 300 dpi
plt.savefig('tested_vs_generated_data_SOC.jpg', dpi=300)
#plt.show()

# Encode all data points
all_data_combined = np.hstack([SOC[:, np.newaxis], Fts])
all_encoded_data = encoder.predict(all_data_combined)
# Visualizing the Latent Space with colors based on SOH
plt.figure(figsize=(4, 4))
scatter = plt.scatter(all_encoded_data[:, 0], all_encoded_data[:, 1], c=data['SOH'], cmap='viridis', s=50)
# Creating and adjusting the colorbar
cbar = plt.colorbar(scatter, label='SOH')
cbar.ax.set_ylabel('SOH', fontsize=size)
cbar.ax.tick_params(labelsize=size)
plt.title("Visualization of the Latent Space", fontsize=size)
plt.xlabel("Latent Dimension 1", fontsize=size)
plt.ylabel("Latent Dimension 2", fontsize=size)
plt.xticks(fontsize=size)  # Adjust the x-axis tick font size
plt.yticks(fontsize=size)  # Adjust the y-axis tick font size
plt.tight_layout()
#plt.show()

# output data
# 1. Calculate the augmented_SOH based on the unique SOH values and the augmented data size.
augmented_SOH = np.concatenate([np.full(shape=(len(data[data['SOH'] == soh]) * sampling_multiplier,), fill_value=soh) for soh in SOH_values])
augmented_data_array = np.column_stack((np.ones(len(augmented_SOH)), augmented_SOH, augmented_SOC, augmented_SOE, augmented_Fts))
# 2. Prepare Original Data
original_data_array = np.column_stack((np.zeros(len(data['SOH'])), data['SOH'].values, data['SOC'].values, data['SOE'].values, data.loc[:, 'U1':'U21'].values))
# 3. Combine both data arrays
combined_data_array = np.vstack((original_data_array, augmented_data_array))
# Convert the combined array to a DataFrame
columns = ['Augmented', 'SOH', 'SOC', 'SOE'] + ['U' + str(i) for i in range(1, 22)]
augmented_df = pd.DataFrame(augmented_data_array, columns=columns)
combined_df = pd.DataFrame(combined_data_array, columns=columns)
# Save the DataFrame to Excel
#combined_df.to_excel("combined_augmented_data_output_Cylind21.xlsx", index=False)
#augmented_df.to_excel("augmented_data_output_Cylind21.xlsx", index=False)

#combined_df.to_excel("combined_augmented_data_output_Pouch31.xlsx", index=False)
#augmented_df.to_excel("augmented_data_output_Pouch31.xlsx", index=False)

#combined_df.to_excel("combined_augmented_data_output_Pouch52.xlsx", index=False)
#augmented_df.to_excel("augmented_data_output_Pouch52.xlsx", index=False)

from scipy.stats import gaussian_kde, entropy
# Define number of bins
n_bins = 50
# Select the first dimension (U1)
original_data = Fts[:, 0]
augmented_data = augmented_Fts[:, 0]
# Plot histograms
plt.figure(figsize=(4, 4))
counts1, bin_edges1 = np.histogram(original_data, bins=n_bins, density=True)
plt.hist(original_data, bins=bin_edges1, alpha=0.5, label='Tested', density=True)
counts2, bin_edges2 = np.histogram(augmented_data, bins=bin_edges1, density=True)
plt.hist(augmented_data, bins=bin_edges1, alpha=0.5, label='Generated', density=True)
# Fit KDE to original and augmented data
kde_orig = gaussian_kde(original_data)
kde_augmented = gaussian_kde(augmented_data)
# Define points for plotting
x_range = np.linspace(min(bin_edges1), max(bin_edges1), 1000)
# Calculate densities
density_orig = kde_orig(x_range)
density_augmented = kde_augmented(x_range)
# Overlay the KDE on histogram
plt.plot(x_range, density_orig, label='KDE-Tested')
plt.plot(x_range, density_augmented, label='KDE-Generated')
# Calculate KL divergence
kl_div = entropy(pk=counts1 + 1e-10, qk=counts2 + 1e-10)  # Adding small constant for numerical stability

# Plot title and labels
#plt.title(f"Feature U1 Distribution - KL Divergence: {kl_div:.4f}")
plt.xlabel('Dimension of Voltage Dynamics U1 [V]')
plt.ylabel("Gaussian Kernel Density")
# y limit from 0 to 10
plt.ylim(0, 10)
plt.xlim(3.4,4.2)
plt.legend()
# save the plot for file_path data to jpg 300 dpi
plt.savefig('tested_vs_generated_U1_distribution.jpg', dpi=300)
plt.show()