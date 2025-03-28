import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
import openpyxl

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
file_path1 = "data/data_Cylind21.xlsx"
file_path2 = "data/data_Pouch31.xlsx"
file_path3 = "data/data_Pouch52.xlsx"
file_path4 = "data/combined_augmented_data_output_Cylind21.xlsx"
file_path5 = "data/combined_augmented_data_output_Pouch31.xlsx"
file_path6 = "data/combined_augmented_data_output_Pouch52.xlsx"

data1 = pd.read_excel(file_path1, sheet_name="All")
data2 = pd.read_excel(file_path2, sheet_name="All")
data3 = pd.read_excel(file_path3, sheet_name="All")
data4 = pd.read_excel(file_path4, sheet_name="Sheet1")
data5 = pd.read_excel(file_path5, sheet_name="Sheet1")
data6 = pd.read_excel(file_path6, sheet_name="Sheet1")

data1['Cata'] = 1
data2['Cata'] = 2
data3['Cata'] = 3
data4['Cata'] = 1
data5['Cata'] = 2
data6['Cata'] = 3

# Concatenation
data = pd.concat([data1, data2, data3], ignore_index=True)
filtered_data = data[(data['Cata'] == 2) | (data['Cata'] == 3)]

data_aug = pd.concat([data4, data5, data6], ignore_index=True)
filtered_data_aug = data_aug[(data_aug['Cata'] == 2) | (data_aug['Cata'] == 3)]

# Apply mask
mask = filtered_data['SOH'] <= 2
Fts = filtered_data.loc[mask, 'U1':'U21'].values
SOH = filtered_data[mask]['SOH'].values
SOC = filtered_data[mask]['SOC'].values
Cata = filtered_data[mask]['Cata'].values

mask = filtered_data_aug['SOH'] <= 2
Fts_aug = filtered_data_aug.loc[mask, 'U1':'U21'].values
SOH_aug = filtered_data_aug[mask]['SOH'].values
Cata_aug = filtered_data_aug[mask]['Cata'].values

# Feature and Label Normalization
feature_scaler = StandardScaler()
label_scaler_SOH = StandardScaler()
feature_scaler_aug = StandardScaler()
label_scaler_SOH_aug = StandardScaler()

Fts = feature_scaler.fit_transform(Fts)
SOH = label_scaler_SOH.fit_transform(SOH.reshape(-1, 1)).flatten()

Fts_aug = feature_scaler_aug.fit_transform(Fts_aug)
SOH_aug = label_scaler_SOH_aug.fit_transform(SOH_aug.reshape(-1, 1)).flatten()

# Stratified Split
X_train, _, y_train, _, Cata_train, _ = train_test_split(
    Fts, SOH, Cata, test_size=0.2, random_state=0, stratify=Cata)

_, X_test, _, y_test, _, Cata_test = train_test_split(
    Fts_aug, SOH_aug, Cata_aug, test_size=0.2, random_state=0, stratify=Cata_aug)

# Define target sizes we want: 105, 70, 52, 42
target_sizes = [105, 70, 52, 42]

# Calculate source domain data length
source_data_length = X_train.shape[0]

# Calculate downsampling rates based on target sizes
downsampling_rates = [size / source_data_length for size in target_sizes]

# Initialize a dictionary to store downsampled datasets
downsampled_data = {}

for target_size, downsampling_rate in zip(target_sizes, downsampling_rates):
    downsampled_indices = np.random.choice(source_data_length, int(source_data_length * downsampling_rate), replace=False)
    X_train_downsampled = X_train[downsampled_indices]
    y_train_downsampled = y_train[downsampled_indices]
    
    downsampled_data[target_size] = (X_train_downsampled, y_train_downsampled)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "SVM": SVR(),
    "k-NN": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gaussian Process": GaussianProcessRegressor(),
    "MLP": tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(21,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
}

# Initialize an empty list to store results
results_list = []

# Train and predict functions
def train_model(model, X_train, y_train):
    if isinstance(model, tf.keras.Sequential):
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=1500, batch_size=512, verbose=0)
    else:
        model.fit(X_train, y_train)

def predict_model(model, X_test):
    if isinstance(model, tf.keras.Sequential):
        return model.predict(X_test).flatten()
    else:
        return model.predict(X_test)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    y_true_inv = label_scaler_SOH.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_inv = label_scaler_SOH.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / y_true_inv)) * 100
    pearson_corr, _ = pearsonr(y_true_inv, y_pred_inv)
    return mape, pearson_corr

# Loop over each model to train and predict on downsampled data
for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    for target_size, (X_train_downsampled, y_train_downsampled) in downsampled_data.items():
        train_model(model, X_train_downsampled, y_train_downsampled)

        for cata_val in [2, 3]:
            X_test_cata = X_test[Cata_test == cata_val]
            y_test_cata = y_test[Cata_test == cata_val]

            y_pred = predict_model(model, X_test_cata)
            mape, pearson_corr = calculate_metrics(y_test_cata, y_pred)

            # Append results to the list
            results_list.append({
                'Model': model_name,
                'Target Size': target_size,
                'Cata': f'Cata{cata_val}',
                'MAPE': mape,
                'Pearson': pearson_corr
            })

# Convert results list to DataFrame
df_results = pd.DataFrame(results_list)

# Save results (MAPE and Pearson correlation) to Excel
file_path = 'model_results.xlsx'

# Ensure file is not corrupted
if os.path.exists(file_path):
    try:
        pd.ExcelFile(file_path)
    except Exception as e:
        os.remove(file_path)

# Create Excel file if it doesn't exist and save the results
with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='Benckmarking-Results', index=False)

print("Model training and evaluation complete. Results saved to Excel.")
