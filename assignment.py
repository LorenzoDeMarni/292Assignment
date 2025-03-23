import pandas as pd
import matplotlib.pyplot as plt

# Load Lorenzo's data
lorenzo_walking_df = pd.read_csv('walking_lorenzo.csv')
lorenzo_jumping_df = pd.read_csv('jumping_lorenzo.csv')

# Load KayKay's data
kaykay_walking_df = pd.read_csv('walking_kaykay.csv')
kaykay_jumping_df = pd.read_csv('jumping_kaykay.csv')

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

# Show all plots
plt.show()