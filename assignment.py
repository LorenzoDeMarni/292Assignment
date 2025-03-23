import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Load Lorenzo's data
lorenzo_walking_df = pd.read_csv('walking_lorenzo.csv')
lorenzo_jumping_df = pd.read_csv('jumping_lorenzo.csv')

# Load KayKay's data
kaykay_walking_df = pd.read_csv('walking_kaykay.csv')
kaykay_jumping_df = pd.read_csv('jumping_kaykay.csv')

# Load Daniil's data 
# daniil_walking_df = pd.read_csv('walking_daniil.csv')
# daniil_jumping_df = pd.read_csv('jumping_daniil.csv')

# Function to compute features for a dataset
def compute_features(df, column_name='Absolute acceleration (m/s^2)'):
    data = df[column_name]
    
    # Compute features
    features = {
        'max': data.max(),
        'min': data.min(),
        'range': data.max() - data.min(),
        'mean': data.mean(),
        'median': data.median(),
        'variance': data.var(),
        'std_dev': data.std(),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'rms': np.sqrt(np.mean(data**2))
    }
    
    return features

# Organize datasets with labels
datasets = {
    'Lorenzo Walking': lorenzo_walking_df,
    'Lorenzo Jumping': lorenzo_jumping_df,
    'KayKay Walking': kaykay_walking_df,
    'KayKay Jumping': kaykay_jumping_df,
    # 'Daniil Walking': daniil_walking_df,
    # 'Daniil Jumping': daniil_jumping_df
}

# Compute features for all datasets
features_dict = {}
for name, df in datasets.items():
    features_dict[name] = compute_features(df)

# Create DataFrame with all features
features_df = pd.DataFrame.from_dict(features_dict, orient='index')
print("Original Features:")
print(features_df)

# Normalize features using z-score standardization
scaler = StandardScaler()
normalized_values = scaler.fit_transform(features_df)
normalized_df = pd.DataFrame(normalized_values, 
                            index=features_df.index, 
                            columns=features_df.columns)
print("\nNormalized Features (Z-score):")
print(normalized_df)

# Figure 1: Lorenzo's data (walking and jumping in one window)
fig_lorenzo, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig_lorenzo.suptitle('Lorenzo Data - Absolute Acceleration', fontsize=16)

# Plot Lorenzo Walking absolute acceleration
ax1.plot(lorenzo_walking_df['Time (s)'], lorenzo_walking_df['Absolute acceleration (m/s^2)'], 'b-', label='Walking')
ax1.set_title('Walking')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Absolute Acceleration (m/s^2)')
ax1.grid(True)
ax1.legend()

# Plot Lorenzo Jumping absolute acceleration
ax2.plot(lorenzo_jumping_df['Time (s)'], lorenzo_jumping_df['Absolute acceleration (m/s^2)'], 'r-', label='Jumping')
ax2.set_title('Jumping')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Absolute Acceleration (m/s^2)')
ax2.grid(True)
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

# Figure 2: KayKay's data (walking and jumping in one window)
fig_kaykay, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
fig_kaykay.suptitle('KayKay Data - Absolute Acceleration', fontsize=16)

# Plot KayKay Walking absolute acceleration
ax3.plot(kaykay_walking_df['Time (s)'], kaykay_walking_df['Absolute acceleration (m/s^2)'], 'b-', label='Walking')
ax3.set_title('Walking')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Absolute Acceleration (m/s^2)')
ax3.grid(True)
ax3.legend()

# Plot KayKay Jumping absolute acceleration
ax4.plot(kaykay_jumping_df['Time (s)'], kaykay_jumping_df['Absolute acceleration (m/s^2)'], 'r-', label='Jumping')
ax4.set_title('Jumping')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Absolute Acceleration (m/s^2)')
ax4.grid(True)
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle


# # Plot KayKay Walking absolute acceleration
# ax3.plot(daniil_walking_df['Time (s)'], daniil_walking_df['Absolute acceleration (m/s^2)'], 'b-', label='Walking')
# ax3.set_title('Walking')
# ax3.set_xlabel('Time (s)')
# ax3.set_ylabel('Absolute Acceleration (m/s^2)')
# ax3.grid(True)
# ax3.legend()
#
# # Plot KayKay Jumping absolute acceleration
# ax4.plot(daniil_jumping_df['Time (s)'], daniil_jumping_df['Absolute acceleration (m/s^2)'], 'r-', label='Jumping')
# ax4.set_title('Jumping')
# ax4.set_xlabel('Time (s)')
# ax4.set_ylabel('Absolute Acceleration (m/s^2)')
# ax4.grid(True)
# ax4.legend()
#
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle


# Create a figure to visualize feature comparison
plt.figure(figsize=(12, 8))
features_to_display = ['max', 'mean', 'std_dev', 'skewness', 'rms']
bar_width = 0.2
index = np.arange(len(features_to_display))

# Plot bars for each dataset
for i, (name, _) in enumerate(datasets.items()):
    values = [features_df.loc[name, feature] for feature in features_to_display]
    plt.bar(index + i * bar_width, values, bar_width, label=name)

plt.xlabel('Features')
plt.ylabel('Value')
plt.title('Feature Comparison Across Datasets')
plt.xticks(index + bar_width * 1.5, features_to_display)
plt.legend()
plt.tight_layout()

# Show all plots
plt.show()
