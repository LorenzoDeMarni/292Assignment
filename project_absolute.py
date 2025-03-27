import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score

#region ---------------------------------CREATE HDF5 FILE-----------------------------------------------
# Create h5py file for organization
with h5py.File("dataset.h5", "w") as f:
    raw_data_group = f.create_group("Raw Data")
    preprocess_data_group = f.create_group("Pre-processed Data")
    segmented_data_group = f.create_group("Segmented Data")
    segmented_train_data_group = segmented_data_group.create_group("Train")
    segmented_test_data_group = segmented_data_group.create_group("Test")
    
    # Create groups for each person
    for member in ["Lorenzo", "Kaykay"]:
        raw_data_group.create_group(member)
        preprocess_data_group.create_group(member)
        
# Create directories for different file types
directories = ['raw', 'processed', 'segmented']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to print missing values and dashes
def check_missing_values(name, df):
    print(f"{name} NaNs:\n{df.isna().sum()}")
    print(f"{name} dashes: {(df == '-').sum().sum()}\n")
#endregion
#region ---------------------------------LOAD RAW DATA AND STORE IN HDF5-----------------------------------------------
# List of datasets
raw_dfs = {
    "lorenzo_walking": pd.read_csv("raw/lorenzo_walking_raw.csv"),
    "lorenzo_jumping": pd.read_csv("raw/lorenzo_jumping_raw.csv"),
    "kaykay_walking": pd.read_csv("raw/kaykay_walking_raw.csv"),
    "kaykay_jumping": pd.read_csv("raw/kaykay_jumping_raw.csv")
}

# Store raw data in HDF5
with h5py.File("dataset.h5", "a") as f:
    for name, df in raw_dfs.items():
        person = name.split('_')[0].capitalize()
        activity = name.split('_')[1]
        # Store raw data
        f[f"Raw Data/{person}/{activity}"] = df.values
#endregion
#region ------------------------------PROCESS RAW DATA AND STORE IN HDF5-----------------------------------------------
# Function to preprocess acceleration data
def preprocess_dataframe(df, window_size=10):
    processed_df = df.copy()
    # Only process absolute acceleration column
    processed_df['Absolute acceleration (m/s^2)'] = df['Absolute acceleration (m/s^2)'].rolling(window=window_size).mean().bfill()
    return processed_df

# Apply preprocessing to all datasets
processed_dfs={}
for name, df in raw_dfs.items():
    processed_dfs[name]=preprocess_dataframe(df)

# Save processed datasets
for name, df in processed_dfs.items():
    file_name = f"{name}_processed.csv"
    df.to_csv(os.path.join('processed', file_name), index=False)
    print(f"✅ Processed data saved: {file_name}")

# Store processed data in HDF5
with h5py.File("dataset.h5", "a") as f:
    for name, df in processed_dfs.items():
        person = name.split('_')[0].capitalize()
        activity = name.split('_')[1]
        # Store processed data
        f[f"Pre-processed Data/{person}/{activity}"] = df.values
#endregion
#region ---------------------------------PLOT DATA (RAW vs FILTERED)-----------------------------------------------
# Figure 1: Lorenzo's data (walking and jumping in one window)
fig_lorenzo, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig_lorenzo.suptitle('Lorenzo Data - Absolute Acceleration Processing', fontsize=16)

# Plot Lorenzo Walking acceleration data
ax1.plot(raw_dfs['lorenzo_walking']['Time (s)'], raw_dfs['lorenzo_walking']['Absolute acceleration (m/s^2)'], 'k-', alpha=0.7, label='Raw Absolute')
ax1.plot(processed_dfs['lorenzo_walking']['Time (s)'], processed_dfs['lorenzo_walking']['Absolute acceleration (m/s^2)'], 'g-', label='Filtered Absolute')
ax1.set_title('Walking')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (m/s²)')
ax1.grid(True)
ax1.legend()

# Plot Lorenzo Jumping acceleration data
ax2.plot(raw_dfs['lorenzo_jumping']['Time (s)'], raw_dfs['lorenzo_jumping']['Absolute acceleration (m/s^2)'], 'k-', alpha=0.7, label='Raw Absolute')
ax2.plot(processed_dfs['lorenzo_jumping']['Time (s)'], processed_dfs['lorenzo_jumping']['Absolute acceleration (m/s^2)'], 'g-', label='Filtered Absolute')
ax2.set_title('Jumping')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Acceleration (m/s²)')
ax2.grid(True)
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for subtitle

# Figure 2: KayKay's data (walking and jumping in one window)
fig_kaykay, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
fig_kaykay.suptitle('KayKay Data - Absolute Acceleration Processing', fontsize=16)

# Plot KayKay Walking acceleration data
ax3.plot(raw_dfs['kaykay_walking']['Time (s)'], raw_dfs['kaykay_walking']['Absolute acceleration (m/s^2)'], 'k-', alpha=0.7, label='Raw Absolute')
ax3.plot(processed_dfs['kaykay_walking']['Time (s)'], processed_dfs['kaykay_walking']['Absolute acceleration (m/s^2)'], 'g-', label='Filtered Absolute')
ax3.set_title('Walking')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration (m/s²)')
ax3.grid(True)
ax3.legend()

# Plot KayKay Jumping acceleration data
ax4.plot(raw_dfs['kaykay_jumping']['Time (s)'], raw_dfs['kaykay_jumping']['Absolute acceleration (m/s^2)'], 'k-', alpha=0.7, label='Raw Absolute')
ax4.plot(processed_dfs['kaykay_jumping']['Time (s)'], processed_dfs['kaykay_jumping']['Absolute acceleration (m/s^2)'], 'g-', label='Filtered Absolute')
ax4.set_title('Jumping')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Acceleration (m/s²)')
ax4.grid(True)
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for subtitle

# Show all plots
plt.show()
#endregion
#region ---------------------------------SEGMENT DATA-----------------------------------------------
#compute average sampling frequency for Lorenzo
lorenzo_sampling_rate = 1 / processed_dfs['lorenzo_walking'].iloc[:, 0].diff().mean()
print(f"Lorenzo Estimated Sampling Frequency: {lorenzo_sampling_rate:.2f} Hz")

#compute average sampling frequency for KayKay
kaykay_sampling_rate = 1 / processed_dfs['kaykay_jumping'].iloc[:, 0].diff().mean()
print(f"KayKay Estimated Sampling Frequency: {kaykay_sampling_rate:.2f} Hz")

if (round(lorenzo_sampling_rate/50)*50) == (round(kaykay_sampling_rate/50)*50):
    official_sample_rate = (round(lorenzo_sampling_rate/50)*50)
    
print(f"Official Sample rate: {official_sample_rate}")

window_size=5*official_sample_rate

def segment_data_5s(data, window_size):     
    segments=[]
    # Only include absolute acceleration column
    for i in range(0, len(data), window_size):
        segment=data.iloc[i:i+window_size, 4].values  # Get only absolute acceleration column
        if len(segment) == window_size:
            segments.append(segment)
    
    return np.array(segments)

segmented_arrays={}
for name, df in processed_dfs.items():
    segmented_arrays[name]=segment_data_5s(df, window_size)

# Print segment shapes to understand their dimensions
print("Segment shapes:")
for name, array in segmented_arrays.items():
    print(f"{name}: {array.shape}")
#endregion
#region ---------------------------------EXTRACT FEATURES-----------------------------------------------
# Define function to calculate zero crossing rate
def zero_crossing_rate(data):
    zcr = np.sum(np.diff(np.signbit(data))) / (len(data) - 1)
    return zcr

# Extract features for each segment (returns a dictionary of features)
def extract_features(segment):
    features = {}
    # Basic features
    features["mean"] = np.mean(segment)
    features["std"] = np.std(segment)
    features["min"] = np.min(segment)
    features["max"] = np.max(segment)
    features["range"] = np.ptp(segment)
    features["variance"] = np.var(segment)
    features["median"] = np.median(segment)
    features["rms"] = np.sqrt(np.mean(np.square(segment)))
    features["zero_crossing_rate"] = zero_crossing_rate(segment)
    features["kurtosis"] = stats.kurtosis(segment)
    features["skewness"] = stats.skew(segment)

    return features

# Extract features for each activity and person
features_arrays={}
for name, array in segmented_arrays.items():
    features_arrays[name]=[]
    for segment in array:
        features_arrays[name].append(extract_features(segment))

# define feature names
feature_names = [
        'mean', 'std', 'min', 'max', 'range', 'variance', 
        'median', 'rms', 'zero_crossing_rate', 'kurtosis', 'skewness'
    ]

# Function to create a DataFrame with clear column names
def features_to_dataframe(features_list):
    # Create column names in the format feature (mean, std, ...)
    columns = feature_names
    
    # Create the data rows
    data = [] # list of lists(rows)
    for features in features_list: # iterates through each 5 second segment with feature data
        row = []
        for feature in feature_names: # adds data for each feature (11 total)
            row.append(features[feature])
        data.append(row)
    
    # Create and return the DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df

# Convert features to DataFrames with clear column names
features_dfs={}
for name, array in features_arrays.items():
    features_dfs[name]=features_to_dataframe(array)
#endregion
#region ---------------------------------NORMALIZE FEATURES-----------------------------------------------

# Normalize features
scaler=StandardScaler()
def normalize_df(df):
    scaled_data=scaler.fit_transform(df)
    print("Normalizing segmented features...")
    return pd.DataFrame(scaled_data, columns=df.columns)

normalized_dfs={}
for name, df in features_dfs.items():
    normalized_dfs[name]=normalize_df(df)

# Save normalized features to CSV files with clear headers
for name, df in normalized_dfs.items():
    df.to_csv(os.path.join('segmented', f'{name}_segmented.csv'), index=False)
    print(f"✅ Segmented data saved: {name}")
#endregion
#region ---------------------------------CREATE FINAL DATASET-----------------------------------------------

# Create final dataset with activity column (0 for walking, 1 for jumping) and concatenate all dataframes
final_dataset = pd.DataFrame()
for name, df in normalized_dfs.items():
    if (name.split('_')[-1]=='walking'):
        df['activity'] = 0
    elif name.split('_')[-1] == 'jumping':
        df['activity'] = 1
    final_dataset = pd.concat([final_dataset, df], axis=0, ignore_index=True)

print(final_dataset.shape)  # (rows, columns)
#endregion
#region ---------------------------------TRAIN LOGISTIC REGRESSION-----------------------------------------------
final_data = final_dataset.iloc[:,:-1]
final_labels = final_dataset.iloc[:,-1]
# 10% of data for testing, 90% for training
X_train, X_test, y_train, y_test = \
    train_test_split(final_data, final_labels, test_size=0.1, shuffle=True, random_state=0) 

l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(l_reg)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
clf_probs = clf.predict_proba(X_test)
acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print(f"Predictions: {predictions}") # 0 for walking, 1 for jumping
print(f"Probabilities: {clf_probs}") # probability of being 0 or 1
print(f"Accuracy: {acc}") # percentage of correct predictions
print(f"Recall: {recall}") # percentage of true positives

cm = confusion_matrix(y_test, predictions)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, clf_probs[:,1], pos_label=clf.classes_[1])
roc_display= RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()
auc = roc_auc_score(y_test, clf_probs[:,1])
print(f"AUC: {auc}")
#endregion
#region ---------------------------------PLOT SEGMENTED DATA----------------------------------------------- 
# Plot first 5-second segments of Lorenzo's and KayKay's segmented data
fig_lorenzo_seg, axes = plt.subplots(2, 2, figsize=(15, 10))
fig_lorenzo_seg.suptitle('Lorenzo & KayKay Data - Segmented Absolute Acceleration', fontsize=16)
# Create a time axis for a single 5-second segment
time_axis = np.linspace(0, 5, int(window_size), endpoint=False)  # From 0 to 5 seconds

# Flatten axes array
ax1, ax2, ax3, ax4 = axes.flatten()

# Plot Lorenzo Walking segmented data
ax1.plot(time_axis, segmented_arrays['lorenzo_walking'][20], 'g-', label='Absolute')
ax1.set_title('Lorenzo Walking')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Acceleration (m/s²)')
ax1.legend()
ax1.grid(True)

# Plot Lorenzo Jumping segmented data
ax2.plot(time_axis, segmented_arrays['lorenzo_jumping'][20], 'g-', label='Absolute')
ax2.set_title('Lorenzo Jumping')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Acceleration (m/s²)')
ax2.legend()
ax2.grid(True)

# Plot KayKay Walking segmented data
ax3.plot(time_axis, segmented_arrays['kaykay_walking'][20], 'g-', label='Absolute')
ax3.set_title('KayKay Walking')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Acceleration (m/s²)')
ax3.legend()
ax3.grid(True)

# Plot KayKay Jumping segmented data
ax4.plot(time_axis, segmented_arrays['kaykay_jumping'][20], 'g-', label='Absolute')
ax4.set_title('KayKay Jumping')
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Acceleration (m/s²)')
ax4.legend()
ax4.grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#endregion  
