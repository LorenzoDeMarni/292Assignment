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
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score)
import joblib

#region ---------------------------------CREATE HDF5 FILE-----------------------------------------------
# Create h5py file for organization
with h5py.File("dataset.h5", "w") as f:
    raw_data_group = f.create_group("Raw Data")
    preprocess_data_group = f.create_group("Pre-processed Data")
    segmented_data_group = f.create_group("Segmented Data")
    segmented_train_data_group = segmented_data_group.create_group("Train")
    segmented_test_data_group = segmented_data_group.create_group("Test")
    
    # Create groups for each person
    for member in ["Lorenzo", "Kaykay", "Daniil"]:
        raw_data_group.create_group(member)
        preprocess_data_group.create_group(member)

# Create directories for different file types
directories = ['raw', 'processed', 'segmented']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_missing_values(name, df):
    """Print missing values and dash-based missing markers."""
    print(f"{name} NaNs:\n{df.isna().sum()}")
    print(f"{name} dashes: {(df == '-').sum().sum()}\n")
#endregion

#region ---------------------------------LOAD RAW DATA AND STORE IN HDF5-----------------------------------------------
# List of datasets. Adjust if you have different filenames.
raw_dfs = {
    "lorenzo_walking": pd.read_csv("raw/lorenzo_walking_raw.csv"),
    "lorenzo_jumping": pd.read_csv("raw/lorenzo_jumping_raw.csv"),
    "kaykay_walking": pd.read_csv("raw/kaykay_walking_raw.csv"),
    "kaykay_jumping": pd.read_csv("raw/kaykay_jumping_raw.csv"),
    "daniil_walking": pd.read_csv("raw/daniil_walking_raw.csv"),
    "daniil_jumping": pd.read_csv("raw/daniil_jumping_raw.csv")
}

with h5py.File("dataset.h5", "a") as f:
    for name, df in raw_dfs.items():
        person = name.split('_')[0].capitalize()
        activity = name.split('_')[1]
        # Store raw data in HDF5
        f[f"Raw Data/{person}/{activity}"] = df.values
#endregion

#region ------------------------------PROCESS RAW DATA AND STORE IN HDF5-----------------------------------------------  
def preprocess_dataframe(df, window_size=51):
    """
    Filter data with rolling mean with bfill.
    Automatically detect time column so we skip it.
    """
    processed_df = df.copy()
    
    
    # Process only non-time columns
    cols_to_process = df.columns[1:]
    
    for col in cols_to_process:
        processed_df[col] = df[col].rolling(window=window_size).mean().bfill()
    return processed_df

# Print missing values
for name, df in raw_dfs.items():
    check_missing_values(name, df)

# Preprocess all datasets
processed_dfs = {}
for name, df in raw_dfs.items():
    processed_dfs[name] = preprocess_dataframe(df)

# Save processed CSVs
for name, df in processed_dfs.items():
    file_name = f"{name}_processed.csv"
    df.to_csv(os.path.join('processed', file_name), index=False)
    print(f"✅ Processed data saved: {file_name}")

# Store processed data in HDF5
with h5py.File("dataset.h5", "a") as f:
    for name, df in processed_dfs.items():
        person = name.split('_')[0].capitalize()
        activity = name.split('_')[1]
        f[f"Pre-processed Data/{person}/{activity}"] = df.values
#endregion

#region ---------------------------------PLOT DATA (RAW vs FILTERED)-----------------------------------------------
fig_lorenzo, axes = plt.subplots(3, 2, figsize=(15, 10))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
fig_lorenzo.suptitle('Acceleration Processing', fontsize=16)

# Lorenzo Walking
time_col = raw_dfs['lorenzo_walking'].columns[0]
abs_col  = raw_dfs['lorenzo_walking'].columns[3]

ax1.plot(raw_dfs['lorenzo_walking'][time_col], raw_dfs['lorenzo_walking'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax1.plot(processed_dfs['lorenzo_walking'][time_col], processed_dfs['lorenzo_walking'][abs_col],'r-', label='Filtered Abs-acceleration')

ax1.set_title('Lorenzo Walking')
ax1.set_xlabel(time_col)
ax1.set_ylabel(abs_col)
ax1.grid(True)
ax1.legend()

# Lorenzo Jumping
time_col = raw_dfs['lorenzo_jumping'].columns[0]
abs_col  = raw_dfs['lorenzo_jumping'].columns[3]

ax2.plot(raw_dfs['lorenzo_jumping'][time_col], raw_dfs['lorenzo_jumping'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax2.plot(processed_dfs['lorenzo_jumping'][time_col], processed_dfs['lorenzo_jumping'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax2.set_title('Lorenzo Jumping')
ax2.set_xlabel(time_col)
ax2.set_ylabel(abs_col)
ax2.grid(True)
ax2.legend()

# Kaykay Walking
time_col = raw_dfs['kaykay_walking'].columns[0]
abs_col  = raw_dfs['kaykay_walking'].columns[3]

ax3.plot(raw_dfs['kaykay_walking'][time_col], raw_dfs['kaykay_walking'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax3.plot(processed_dfs['kaykay_walking'][time_col], processed_dfs['kaykay_walking'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax3.set_title('KayKay Walking')
ax3.set_xlabel(time_col)
ax3.set_ylabel(abs_col)
ax3.grid(True)
ax3.legend()

# Kaykay Jumping
time_col = raw_dfs['kaykay_jumping'].columns[0]
abs_col  = raw_dfs['kaykay_jumping'].columns[3]

ax4.plot(raw_dfs['kaykay_jumping'][time_col], raw_dfs['kaykay_jumping'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax4.plot(processed_dfs['kaykay_jumping'][time_col], processed_dfs['kaykay_jumping'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax4.set_title('KayKay Jumping')
ax4.set_xlabel(time_col)
ax4.set_ylabel(abs_col)
ax4.grid(True)
ax4.legend()

# Daniil Walking
time_col = raw_dfs['daniil_walking'].columns[0]
abs_col  = raw_dfs['daniil_walking'].columns[3]

ax5.plot(raw_dfs['daniil_walking'][time_col], raw_dfs['daniil_walking'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax5.plot(processed_dfs['daniil_walking'][time_col], processed_dfs['daniil_walking'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax5.set_title('Daniil Walking')
ax5.set_xlabel(time_col)
ax5.set_ylabel(abs_col)
ax5.grid(True)
ax5.legend()

# Daniil Jumping
time_col = raw_dfs['daniil_jumping'].columns[0]
abs_col  = raw_dfs['daniil_jumping'].columns[3]

ax6.plot(raw_dfs['daniil_jumping'][time_col], raw_dfs['daniil_jumping'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')    
ax6.plot(processed_dfs['daniil_jumping'][time_col], processed_dfs['daniil_jumping'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax6.set_title('Daniil Jumping')
ax6.set_xlabel(time_col)
ax6.set_ylabel(abs_col)
ax6.grid(True)
ax6.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#endregion

#region ---------------------------------SEGMENT DATA-----------------------------------------------
def segment_data_5s(data, window_size):
    """
    Splits the data into segments of length 'window_size'.
    Skips the time column automatically.
    """
    segments = []

    # Identify time col to skip it
    time_col = data.columns[0]
    
    # Use everything except time col
    data_no_time = data.drop(columns=[time_col])

    for i in range(0, len(data_no_time), window_size):
        segment = data_no_time.iloc[i : i + window_size, :].values
        if len(segment) == window_size:
            segments.append(segment)
    return np.array(segments)

#compute average sampling frequency
lorenzo_sampling_rate = 1 / processed_dfs['lorenzo_walking'][time_col].diff().mean()
kaykay_sampling_rate = 1 / processed_dfs['kaykay_jumping'][time_col].diff().mean()
daniil_sampling_rate = 1 / processed_dfs['daniil_jumping'][time_col].diff().mean()

print(f"Lorenzo Estimated Sampling Frequency: {lorenzo_sampling_rate:.2f} Hz")
print(f"KayKay Estimated Sampling Frequency: {kaykay_sampling_rate:.2f} Hz")
print(f"Daniil Estimated Sampling Frequency: {daniil_sampling_rate:.2f} Hz")

if (round(lorenzo_sampling_rate/50)*50) == (round(kaykay_sampling_rate/50)*50) == (round(daniil_sampling_rate/50)*50):
    official_sample_rate = (round(lorenzo_sampling_rate/50)*50)
else:
    print("Sampling rates are not the same")

print(f"Official Sample rate: {official_sample_rate}")
window_size = int(5 * official_sample_rate)

segmented_arrays = {}
for name, df in processed_dfs.items():
    segmented_arrays[name] = segment_data_5s(df, window_size)

print("Segment shapes:")
for name, array in segmented_arrays.items():
    print(f"{name}: {array.shape}")
#endregion

#region ---------------------------------PLOT SEGMENTED DATA EXAMPLE (Second Segment)-----------------------------------------------
# Example: Plot second 5-second segments for all users walking vs jumping
# In 'segmented_arrays', columns: 0->x,1->y,2->z,3->abs
time_axis = np.linspace(0, 5, int(window_size), endpoint=False)

fig_lorenzo_seg, axes = plt.subplots(1, 2, figsize=(15, 5))
fig_lorenzo_seg.suptitle('Segmented Data Example (2nd 5s segment)', fontsize=16)
ax1, ax2 = axes.flatten()

# Lorenzo walking
lorenzo_walk = segmented_arrays['lorenzo_walking'][2]
kaykay_walk = segmented_arrays['kaykay_walking'][2]
daniil_walk = segmented_arrays['daniil_walking'][2] 

ax1.plot(time_axis, lorenzo_walk[:, 3], 'k-', label='Lorenzo')
ax1.plot(time_axis, kaykay_walk[:, 3], 'b-', label='KayKay')
ax1.plot(time_axis, daniil_walk[:, 3], 'r-', label='Daniil')
ax1.set_title('Walking Absolute Acceleration')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (m/s²)')
ax1.legend()
ax1.grid(True)

# Lorenzo jumping
lorenzo_jump = segmented_arrays['lorenzo_jumping'][2]  
kaykay_jump = segmented_arrays['kaykay_jumping'][2]  
daniil_jump = segmented_arrays['daniil_jumping'][2]  

ax2.plot(time_axis, lorenzo_jump[:, 3], 'k-', label='Lorenzo')
ax2.plot(time_axis, kaykay_jump[:, 3], 'b-', label='KayKay')
ax2.plot(time_axis, daniil_jump[:, 3], 'r-', label='Daniil')
ax2.set_title('Jumping Absolute Acceleration')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Acceleration (m/s²)')
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#endregion

#region ---------------------------------EXTRACT FEATURES-----------------------------------------------
def extract_features(segment):
    '''
    Extract features from a segment of data.
    '''
    features = {}
    features["mean"] = np.mean(segment, axis=0)
    features["std"] = np.std(segment, axis=0)
    features["min"] = np.min(segment, axis=0)
    features["max"] = np.max(segment, axis=0)
    features["range"] = np.ptp(segment, axis=0)
    features["variance"] = np.var(segment, axis=0)
    features["median"] = np.median(segment, axis=0)
    features["rms"] = np.sqrt(np.mean(np.square(segment), axis=0))
    features["kurtosis"] = stats.kurtosis(segment, axis=0)
    features["skewness"] = stats.skew(segment, axis=0)
    return features

feature_names = [
    'mean', 'std', 'min', 'max', 'range', 'variance',
    'median', 'rms', 'kurtosis', 'skewness'
]

def features_to_dataframe(features_list):
    '''
    Convert features list to a dataframe.
    '''
    axes = ['x', 'y', 'z', 'abs']
    
    columns = []
    for feature in feature_names:
        for axis in axes:
            columns.append(f"{feature}_{axis}")
    
    data_rows = []
    for feat_dict in features_list:
        row = []
        for feature in feature_names:
            row.extend(feat_dict[feature])
        data_rows.append(row)
    
    return pd.DataFrame(data_rows, columns=columns)

# Extract features from each segment
features_arrays = {}
for name, array in segmented_arrays.items():
    feats_for_name = []
    for segment in array:
        feats_for_name.append(extract_features(segment))
    features_arrays[name] = feats_for_name

# Convert features list to a dataframe
features_dfs = {}
for name, feat_list in features_arrays.items():
    features_dfs[name] = features_to_dataframe(feat_list)

# Save the feature DataFrame to CSV
for name, df in features_dfs.items():
    out_file = os.path.join('segmented', f'{name}_segmented.csv')
    df.to_csv(out_file, index=False)
    print(f"✅ Segmented data saved: {out_file}")
#endregion

#region ---------------------------------CREATE FINAL DATASET-----------------------------------------------
final_dataset = pd.DataFrame()
for name, df in features_dfs.items():
    activity_label = 0 if name.endswith('walking') else 1  # If not walking, assume jumping
    df['activity'] = activity_label
    final_dataset = pd.concat([final_dataset, df], axis=0, ignore_index=True)

print(f"Final dataset shape: {final_dataset.shape}")  # (rows, columns)
#endregion

#region ---------------------------------SPECIFY WHICH AXIS TO TRAIN WITH-----------------------------------------------
# Filter columns to only include abs-acceleration related features

# Exclude the 'activity' column from feature extraction
feature_columns = final_dataset.columns[:-1]
num_axes = 4  # x, y, z, abs

# Use modulus to extract each axis's features
x_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 0]
y_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 1]
z_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 2]
abs_cols = [col for i, col in enumerate(feature_columns) if i % num_axes == 3]

print("X features:", x_cols)
print("Y features:", y_cols)
print("Z features:", z_cols)
print("Abs features:", abs_cols)

final_dataset = final_dataset[x_cols + y_cols + z_cols + abs_cols + ['activity']]
#endregion

#region ---------------------------------TRAIN LOGISTIC REGRESSION-----------------------------------------------
# Separate features (only abs) and labels
final_data = final_dataset.drop('activity', axis=1)
final_labels = final_dataset['activity']

# 10% test, 90% train
X_train, X_test, y_train, y_test = train_test_split(
    final_data, final_labels, test_size=0.1, shuffle=True, random_state=0
)

with h5py.File("dataset.h5", "a") as f:
    f["Segmented Data/Train/X"] = X_train.values
    f["Segmented Data/Train/y"] = y_train.values
    
    f["Segmented Data/Test/X"] = X_test.values
    f["Segmented Data/Test/y"] = y_test.values
    
    f["Segmented Data/feature_names"] = list(final_data.columns)
    f["Segmented Data/num_features"] = len(final_data.columns)

print(f"Number of features (columns) being used: {len(final_data.columns)}")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

#train and normalize data using pipeline and standard scaler
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)
clf.fit(X_train, y_train)

joblib.dump(clf, "activity_classifier.pkl")

predictions = clf.predict(X_test)
clf_probs = clf.predict_proba(X_test)
acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f"Predictions: {predictions}")
print(f"Probabilities: {clf_probs}")
print(f"Accuracy: {acc}")
print(f"Recall: {recall}")

cm = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, clf_probs[:, 1], pos_label=clf.classes_[1])
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.title('ROC Curve')
plt.show()
auc = roc_auc_score(y_test, clf_probs[:, 1])
print(f"AUC: {auc}\n")

# Correlation of final abs-only dataset
correlation = final_dataset.corr()['activity'].sort_values(ascending=False)
print(correlation)
#endregion

#region ---------------------------------PLOT CORRELATION BAR GRAPH-----------------------------------------------
fig_correlation, ax = plt.subplots(figsize=(15, 8))
ax.bar(correlation.index, correlation.values)

ax.set_title('Correlation Between Features and Activity', fontsize=14)
ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('Correlation', fontsize=12)

plt.xticks(rotation=90, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
#endregion



joblib.dump(clf, 'activity_classifier.pkl')
print("✅Model saved as activity_classifier.pkl")
