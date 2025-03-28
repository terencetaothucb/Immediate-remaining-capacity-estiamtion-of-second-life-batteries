import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import mse
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy

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

# Train VAE and generate augmented data
for soh in SOH_values:
    mask = data['SOH'] == soh
    current_Fts = Fts[mask]
    current_SOC = SOC[mask][:, np.newaxis]
    combined_data = np.hstack([current_SOC, current_Fts])
    scaler = MinMaxScaler().fit(combined_data)
    combined_data_normalized = scaler.transform(combined_data)
    vae.fit(combined_data_normalized, epochs=epochs, batch_size=batch_size, verbose=0)

    num_samples = len(combined_data_normalized) * sampling_multiplier
    random_latent_values = K.random_normal(shape=(num_samples, latent_dim), seed=0)
    new_data_samples_normalized = decoder.predict(random_latent_values)
    new_data_samples = scaler.inverse_transform(new_data_samples_normalized)
    augmented_data.append(new_data_samples)

    augmented_SOC_current = new_data_samples[:, 0]
    augmented_SOE_current = augmented_SOC_current * soh   
    augmented_SOE_list.append(augmented_SOE_current)

# Combine all augmented data
all_augmented_data = np.vstack(augmented_data)
augmented_Fts = all_augmented_data[:, 1:]

# Function to calculate KL divergence for each feature
def calculate_kl_divergences(Fts, augmented_Fts, n_bins=50):
    kl_divergences = []
    for i in range(Fts.shape[1]):
        original_data = Fts[:, i]
        augmented_data = augmented_Fts[:, i]

        # Calculate histograms
        counts1, bin_edges1 = np.histogram(original_data, bins=n_bins, density=True)
        counts2, bin_edges2 = np.histogram(augmented_data, bins=bin_edges1, density=True)

        # Calculate KL divergence
        kl_div = entropy(pk=counts1 + 1e-10, qk=counts2 + 1e-10)  # Adding small constant for numerical stability
        kl_divergences.append(kl_div)
    return kl_divergences

# Calculate KL divergences for all features (U1 to U21)
kl_divergences = calculate_kl_divergences(Fts, augmented_Fts)

plt.figure(figsize=(4, 4))
plt.bar([f'U{i+1}' for i in range(Fts.shape[1])], kl_divergences, color='gray', edgecolor='black')
plt.ylabel('KL Divergence', fontsize=10)
# region of y value 0 to 0.7
plt.ylim(0, 1)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('kl_divergences_bar_chart.jpg', dpi=300)
plt.show()

# Output KL divergence values
for i, kl in enumerate(kl_divergences):
    print(f"KL Divergence for U{i+1}: {kl:.4f}")
