from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras import layers

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
file_path1 = 'data/combined_augmented_data_output_Cylind21.xlsx'
file_path2 = 'data/combined_augmented_data_output_Pouch31.xlsx'
file_path3 = 'data/combined_augmented_data_output_Pouch52.xlsx'
data1 = pd.read_excel(file_path1, sheet_name="Sheet1")
data2 = pd.read_excel(file_path2, sheet_name="Sheet1")
data3 = pd.read_excel(file_path3, sheet_name="Sheet1")

data1['Cata'] = 1 #Cylind21
data2['Cata'] = 2 #Pouch31
data3['Cata'] = 3 #Pouch52

# Concatenation
data = pd.concat([data1, data2, data3], ignore_index=True)

Cata_to_test = 2
sample_proportion = 3 # 1/n th of Cata1 for data availability simulation

# Create a mask where SOH values are less than or equal to 0.95
if Cata_to_test == 3:
    mask = data['SOH'] <= 2
elif Cata_to_test == 2:
    mask = data['SOH'] <= 0.95

# Apply the mask to the different data arrays
Fts = data.loc[mask, 'U1':'U21'].values
SOH = data[mask]['SOH'].values
SOC = data[mask]['SOC'].values
Cata = data[mask]['Cata'].values

# Feature and Label Normalization
feature_scaler = StandardScaler()
label_scaler_SOH = StandardScaler()
SOC_scaler = StandardScaler()

Fts = feature_scaler.fit_transform(Fts)
SOH = label_scaler_SOH.fit_transform(SOH.reshape(-1, 1)).flatten()
SOC = SOC_scaler.fit_transform(SOC.reshape(-1, 1)).flatten()

# Stratified Split
X_train, X_test, y_train, y_test, SOC_train, SOC_test, Cata_train, Cata_test = train_test_split(
    Fts, SOH, SOC, Cata, test_size=0.2, random_state=0, stratify=Cata)

# Filter data for Cata=1 for training, source domain
X_train_Cata1 = X_train[Cata_train == 1]
y_train_Cata1 = y_train[Cata_train == 1]
SOC_train_Cata1 = SOC_train[Cata_train == 1]
# Filter data for Cata=1 for testing, source domain
X_test_Cata1 = X_test[Cata_test == 1]
y_test_Cata1 = y_test[Cata_test == 1]
SOC_test_Cata1 = SOC_test[Cata_test == 1]
# Filter data for Cata=2 for training, target domain
X_train_Cata2 = X_train[Cata_train == Cata_to_test]
y_train_Cata2 = y_train[Cata_train == Cata_to_test]
SOC_train_Cata2 = SOC_train[Cata_train == Cata_to_test] 

# Extend Feature Matrix to include SOC
SOC_train_Cata1 = SOC_train_Cata1.reshape(-1, 1)
SOC_train_Cata2 = SOC_train_Cata2.reshape(-1, 1)

#X_train_Cata1 = np.hstack((X_train_Cata1, SOC_train_Cata1))
#X_train_Cata2 = np.hstack((X_train_Cata2, SOC_train_Cata2))

# Randomly downsample Cata2 to make its size 1/n th of Cata1
sample_size_Cata2 = len(X_train_Cata1) // sample_proportion
random_indices = np.random.choice(len(X_train_Cata2), sample_size_Cata2, replace=False)
X_train_Cata2 = X_train_Cata2[random_indices]
y_train_Cata2 = y_train_Cata2[random_indices]
SOC_train_Cata2 = SOC_train_Cata2[random_indices]
print("The size of target domain training data is:", len(X_train_Cata2))
# Filter data for Cata=2 for testing, target domain
X_test_Cata2 = X_test[Cata_test == Cata_to_test]
y_test_Cata2 = y_test[Cata_test == Cata_to_test]
SOC_test_Cata2 = SOC_test[Cata_test == Cata_to_test]
SOC_test_Cata1 = SOC_test_Cata1.reshape(-1, 1)
SOC_test_Cata2 = SOC_test_Cata2.reshape(-1, 1)

#X_test_Cata1 = np.hstack((X_test_Cata1, SOC_test_Cata1))
#X_test_Cata2 = np.hstack((X_test_Cata2, SOC_test_Cata2))

def coral_loss(source, target):
    source_coral = tf.matmul(tf.transpose(source), source)
    target_coral = tf.matmul(tf.transpose(target), target)
    loss = tf.reduce_mean(tf.square(source_coral - target_coral))
    return loss

# Define soc estimator model
soc_estimator = tf.keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(21,)), 
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])

# Define feature extractor model
feature_extractor = tf.keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu')
])

# Define regression model
task_net = tf.keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Define Coral Model
class CoralModel(tf.keras.Model):
    def __init__(self, soc_estimator, feature_extractor, task_net, **kwargs):
        super(CoralModel, self).__init__(**kwargs)
        self.soc_estimator = soc_estimator
        self.feature_extractor = feature_extractor
        self.task_net = task_net

    def call(self, Fts):
        Predicted_SOC = self.soc_estimator(Fts)
        extracted_features = self.feature_extractor(Fts)
        Extracted_features_Predicted_SOC = tf.concat([extracted_features, Predicted_SOC], axis=1)
        pred_soh = self.task_net(Extracted_features_Predicted_SOC)
        return pred_soh

    def compile(self, optimizer):
        super(CoralModel, self).compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x_source, soc_source, y_source, x_target, soc_target, y_target = data

        with tf.GradientTape() as tape:
            # Predict SOC values using soc_estimator
            soc_pred_source = self.soc_estimator(x_source)
            soc_pred_target = self.soc_estimator(x_target)
            # Add SOC MSE loss
            soc_loss_source = tf.keras.losses.MeanSquaredError()(soc_source, soc_pred_source)
            soc_loss_target = tf.keras.losses.MeanSquaredError()(soc_target, soc_pred_target)

            source_features = self.feature_extractor(x_source)
            target_features = self.feature_extractor(x_target)
        
            coral = coral_loss(source_features, target_features)
            preds_source = self(x_source, training=True)
            preds_target = self(x_target, training=True)
            
            task_loss_source = tf.keras.losses.MeanSquaredError()(y_source, preds_source)
            task_loss_target = tf.keras.losses.MeanSquaredError()(y_target, preds_target)
            
            total_loss =  (2.5 * soc_loss_target + 3 * task_loss_target + 1.5 * soc_loss_source + 2.5 * task_loss_source) * 0.075   + coral

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss, 
            "task_loss_source": task_loss_source, 
            "task_loss_target": task_loss_target, 
            "coral_loss": coral, 
            "soc_loss_source": soc_loss_source, 
            "soc_loss_target": soc_loss_target   
        }

# Compile model
model = CoralModel(soc_estimator, feature_extractor, task_net)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer)

# Prepare data
dataset_source = tf.data.Dataset.from_tensor_slices((X_train_Cata1, SOC_train_Cata1, y_train_Cata1)).batch(128)
dataset_target = tf.data.Dataset.from_tensor_slices((X_train_Cata2, SOC_train_Cata2, y_train_Cata2)).batch(128)
dataset_combined = tf.data.Dataset.zip((dataset_source, dataset_target))

total_loss_list = []
task_loss_source_list = []
task_loss_target_list = []
coral_loss_list = [] 
soc_loss_source_list = []
soc_loss_target_list = []
# Train model
for epoch in range(5000):
    for batch in dataset_combined:
        data_source, data_target = batch
        x_source, soc_source, y_source = data_source
        x_target, soc_target, y_target = data_target
        loss_metrics = model.train_step((x_source, soc_source, y_source, x_target, soc_target, y_target))
    
    print(f"Epoch {epoch}: Loss = {loss_metrics['loss']}, SOC Loss Source = {loss_metrics['soc_loss_source']}, SOC Loss Target = {loss_metrics['soc_loss_target']}, Task Loss Source = {loss_metrics['task_loss_source']}, Task Loss Target = {loss_metrics['task_loss_target']}, Coral Loss = {loss_metrics['coral_loss']}")

    # Append the losses
    total_loss_list.append(loss_metrics['loss'])
    task_loss_source_list.append(loss_metrics['task_loss_source'])
    task_loss_target_list.append(loss_metrics['task_loss_target'])
    coral_loss_list.append(loss_metrics['coral_loss'])
    soc_loss_source_list.append(loss_metrics['soc_loss_source'])
    soc_loss_target_list.append(loss_metrics['soc_loss_target'])

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_elements = y_true != 0
    mape = np.abs((y_true[nonzero_elements] - y_pred[nonzero_elements]) / y_true[nonzero_elements]).mean() * 100
    return mape
def calculate_maxpe(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_elements = y_true != 0
    maxpe = np.abs((y_true[nonzero_elements] - y_pred[nonzero_elements]) / y_true[nonzero_elements]).max() * 100
    return maxpe

# Predict on the source domain and evaluate
y_pred_source = model.predict(X_test_Cata1)
y_pred_source_inv = label_scaler_SOH.inverse_transform(y_pred_source)
y_test_source_inv = label_scaler_SOH.inverse_transform(y_test_Cata1.reshape(-1, 1))
mape_source = calculate_mape(y_test_source_inv, y_pred_source_inv)
maxpe_source = calculate_maxpe(y_test_source_inv, y_pred_source_inv)
print(f'Source Domain: MAPE = {mape_source}, MaxPE = {maxpe_source}')
# Predict on the target domain and evaluate
y_pred_target = model.predict(X_test_Cata2)
y_pred_target_inv = label_scaler_SOH.inverse_transform(y_pred_target)
y_test_target_inv = label_scaler_SOH.inverse_transform(y_test_Cata2.reshape(-1, 1))
mape_target = calculate_mape(y_test_target_inv, y_pred_target_inv)
maxpe_target = calculate_maxpe(y_test_target_inv, y_pred_target_inv)
print(f'Target Domain: MAPE = {mape_target}, MaxPE = {maxpe_target}')

from scipy.stats import pearsonr

# Calculate and print Pearson correlation for source domain
pearson_corr_source, _ = pearsonr(y_test_source_inv.flatten(), y_pred_source_inv.flatten())
print(f'Source Domain: Pearson Correlation = {pearson_corr_source}')
# Calculate and print Pearson correlation for target domain
pearson_corr_target, _ = pearsonr(y_test_target_inv.flatten(), y_pred_target_inv.flatten())
print(f'Target Domain: Pearson Correlation = {pearson_corr_target}')

# Function to create a parity plot
def plot_parity(true_values, predicted_values, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predicted_values, c='blue', label='Data', alpha=0.3)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    
    # Draw parity line
    min_val = min(min(true_values), min(predicted_values))
    max_val = max(max(true_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parity Line')
    
    plt.legend()

# Create parity plot for source domain
plot_parity(y_test_source_inv.flatten(), y_pred_source_inv.flatten(), 'Parity Plot: Source Domain')
# Create parity plot for target domain
plot_parity(y_test_target_inv.flatten(), y_pred_target_inv.flatten(), 'Parity Plot: Target Domain')

# Predict on the source domain for SOC and evaluate
SOC_pred_source = model.soc_estimator.predict(X_test_Cata1)
SOC_pred_source_inv = SOC_scaler.inverse_transform(SOC_pred_source)
SOC_test_source_inv = SOC_scaler.inverse_transform(SOC_test_Cata1.reshape(-1, 1))
mape_soc_source = calculate_mape(SOC_test_source_inv, SOC_pred_source_inv)
maxpe_soc_source = calculate_maxpe(SOC_test_source_inv, SOC_pred_source_inv)
print(f'Source Domain SOC: MAPE = {mape_soc_source}, MaxPE = {maxpe_soc_source}')

# Predict on the target domain for SOC and evaluate
SOC_pred_target = model.soc_estimator.predict(X_test_Cata2)
SOC_pred_target_inv = SOC_scaler.inverse_transform(SOC_pred_target)
SOC_test_target_inv = SOC_scaler.inverse_transform(SOC_test_Cata2.reshape(-1, 1))
mape_soc_target = calculate_mape(SOC_test_target_inv, SOC_pred_target_inv)
maxpe_soc_target = calculate_maxpe(SOC_test_target_inv, SOC_pred_target_inv)
print(f'Target Domain SOC: MAPE = {mape_soc_target}, MaxPE = {maxpe_soc_target}')

# Calculate and print Pearson correlation for SOC in the source domain
pearson_corr_soc_source, _ = pearsonr(SOC_test_source_inv.flatten(), SOC_pred_source_inv.flatten())
print(f'Source Domain SOC: Pearson Correlation = {pearson_corr_soc_source}')

# Calculate and print Pearson correlation for SOC in the target domain
pearson_corr_soc_target, _ = pearsonr(SOC_test_target_inv.flatten(), SOC_pred_target_inv.flatten())
print(f'Target Domain SOC: Pearson Correlation = {pearson_corr_soc_target}')

# Create parity plots for SOC
plot_parity(SOC_test_source_inv.flatten(), SOC_pred_source_inv.flatten(), 'Parity Plot: Source Domain SOC')
plot_parity(SOC_test_target_inv.flatten(), SOC_pred_target_inv.flatten(), 'Parity Plot: Target Domain SOC')

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(total_loss_list, label='Total Loss')
plt.plot(task_loss_source_list, label='Task Loss Source')
plt.plot(task_loss_target_list, label='Task Loss Target')
plt.plot(coral_loss_list, label='Coral Loss')
plt.plot(soc_loss_source_list, label='SOC Loss Source')
plt.plot(soc_loss_target_list, label='SOC Loss Target')

plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Extract features from source and target domains after training
# Extract features
source_features_transformed = model.feature_extractor.predict(X_test_Cata1)
target_features_transformed = model.feature_extractor.predict(X_test_Cata2)
import seaborn as sns
# Let's visualize the first feature for illustration; repeat for others as needed
FtsID = 0
sns.kdeplot(source_features_transformed[:, FtsID], fill=True, color='blue', label="Source (Transformed)")
#sns.kdeplot(X_test_Cata1[:, FtsID], fill=True, color='blue', label="Source (Original)")
sns.kdeplot(target_features_transformed[:, FtsID], fill=True, color='red', label="Target (Transformed)")
#sns.kdeplot(X_test_Cata2[:, FtsID], fill=True, color='blue', label="Target (Original)")
plt.xlabel('Feature Value')
plt.ylabel('Density')
plt.title('Density of First Feature: Source vs. Target')
plt.legend()
plt.show()
