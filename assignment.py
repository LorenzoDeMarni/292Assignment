import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Load Lorenzo's data
lorenzo_walking_df = pd.read_csv('walking_lorenzo.csv')
lorenzo_jumping_df = pd.read_csv('jumping_lorenzo.csv')

# Load KayKay's data
kaykay_walking_df = pd.read_csv('walking_kaykay.csv')
kaykay_jumping_df = pd.read_csv('jumping_kaykay.csv')

#-----------------------------------------
# Identify missing values
#print(lorenzo_walking_df.isna().sum())  # Count NaNs
# print((lorenzo_walking_df == "-").sum().sum())  # Count dashes

# Identify missing values
# print(kaykay_walking_df.isna().sum())  # Count NaNs
# print((kaykay_walking_df == "-").sum().sum())  # Count dashes


# Create new DataFrames with only Time and filtered Absolute acceleration columns
# Lorenzo's data
lorenzo_walking_processed_df = pd.DataFrame()
lorenzo_walking_processed_df['Time (s)'] = lorenzo_walking_df['Time (s)']
lorenzo_walking_processed_df['Absolute acceleration (m/s^2)'] = lorenzo_walking_df['Absolute acceleration (m/s^2)'].rolling(window=51).mean().bfill()

lorenzo_jumping_processed_df = pd.DataFrame()
lorenzo_jumping_processed_df['Time (s)'] = lorenzo_jumping_df['Time (s)']
lorenzo_jumping_processed_df['Absolute acceleration (m/s^2)'] = lorenzo_jumping_df['Absolute acceleration (m/s^2)'].rolling(window=51).mean().bfill()

# KayKay's data
kaykay_walking_processed_df = pd.DataFrame()
kaykay_walking_processed_df['Time (s)'] = kaykay_walking_df['Time (s)']
kaykay_walking_processed_df['Absolute acceleration (m/s^2)'] = kaykay_walking_df['Absolute acceleration (m/s^2)'].rolling(window=51).mean().bfill()

kaykay_jumping_processed_df = pd.DataFrame()
kaykay_jumping_processed_df['Time (s)'] = kaykay_jumping_df['Time (s)']
kaykay_jumping_processed_df['Absolute acceleration (m/s^2)'] = kaykay_jumping_df['Absolute acceleration (m/s^2)'].rolling(window=51).mean().bfill()

# Save the preprocessed data to new CSV files
lorenzo_walking_processed_df.to_csv('walking_lorenzo_processed.csv', index=False)
lorenzo_jumping_processed_df.to_csv('jumping_lorenzo_processed.csv', index=False)
kaykay_walking_processed_df.to_csv('walking_kaykay_processed.csv', index=False)
kaykay_jumping_processed_df.to_csv('jumping_kaykay_processed.csv', index=False)

#-----------------------------------------

# Figure 1: Lorenzo's data (walking and jumping in one window)
fig_lorenzo, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig_lorenzo.suptitle('Lorenzo Data - Acceleration Processing', fontsize=16)

# Plot Lorenzo Walking acceleration data
ax1.plot(lorenzo_walking_df['Time (s)'], lorenzo_walking_df['Absolute acceleration (m/s^2)'], 'k-', alpha=0.7, label='Original Walking')
ax1.plot(lorenzo_walking_processed_df['Time (s)'], lorenzo_walking_processed_df['Absolute acceleration (m/s^2)'], 'g-', label='Filtered Walking')
ax1.set_title('Walking')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration')
ax1.grid(True)
ax1.legend()

# Plot Lorenzo Jumping acceleration data
ax2.plot(lorenzo_jumping_df['Time (s)'], lorenzo_jumping_df['Absolute acceleration (m/s^2)'], 'k-', alpha=0.7, label='Original Jumping')
ax2.plot(lorenzo_jumping_processed_df['Time (s)'], lorenzo_jumping_processed_df['Absolute acceleration (m/s^2)'], 'g-', label='Filtered Jumping')
ax2.set_title('Jumping')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Acceleration')
ax2.grid(True)
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for subtitle

# Figure 2: KayKay's data (walking and jumping in one window)
fig_kaykay, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
fig_kaykay.suptitle('KayKay Data - Acceleration Processing', fontsize=16)

# Plot KayKay Walking acceleration data
ax3.plot(kaykay_walking_df['Time (s)'], kaykay_walking_df['Absolute acceleration (m/s^2)'], 'k-', alpha=0.7, label='Original Walking')
ax3.plot(kaykay_walking_processed_df['Time (s)'], kaykay_walking_processed_df['Absolute acceleration (m/s^2)'], 'g-', label='Filtered Walking')
ax3.set_title('Walking')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration')
ax3.grid(True)
ax3.legend()

# Plot KayKay Jumping acceleration data
ax4.plot(kaykay_jumping_df['Time (s)'], kaykay_jumping_df['Absolute acceleration (m/s^2)'], 'k-', alpha=0.7, label='Original Jumping')
ax4.plot(kaykay_jumping_processed_df['Time (s)'], kaykay_jumping_processed_df['Absolute acceleration (m/s^2)'], 'g-', label='Filtered Jumping')
ax4.set_title('Jumping')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Acceleration')
ax4.grid(True)
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for subtitle

# Show all plots
plt.show()
