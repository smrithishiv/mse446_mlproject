import matplotlib.pyplot as plt
import numpy as np

# Data
years = [2000, 2004, 2008, 2012, 2016, 2020]
voter_turnout = [54.7, 58.3, 58.2, 56.5, 56.0, 61.3]
age_18_24 = [32.3, 41.9, 44.3, 38, 39.4, 48]
age_25_44 = [49.8, 52.2, 51.9, 49.5, 49, 55]
age_45_64 = [64.1, 66.6, 65, 63.4, 61.7, 65.5]
age_65_plus = [56.4, 60.3, 59.6, 57.6, 58.2, 63.7]

# Colors for bar chart
colors = ['brown', 'purple', 'red', 'green', 'orange', 'blue']

# --- Plot 1: Voter Turnout Over Time ---
plt.figure(figsize=(8, 5))
plt.bar(years, voter_turnout, color=colors)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Voter Turnout (%)', fontsize=12)
plt.title('Voter Turnout Over Time', fontsize=14, fontweight='bold')
plt.ylim(50, 65)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(years)
plt.show()

# --- Plot 2: Voter Turnout by Age Group Over Time ---
plt.figure(figsize=(8, 5))
plt.plot(years, age_18_24, marker='o', label='18-24', color='gold')
plt.plot(years, age_25_44, marker='o', label='25-44', color='orange')
plt.plot(years, age_45_64, marker='o', label='45-64', color='red')
plt.plot(years, age_65_plus, marker='o', label='65+', color='purple')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Voter Turnout (%)', fontsize=12)
plt.title('Voter Turnout by Age Group Over Time (2000-2020)', fontsize=14, fontweight='bold')
plt.ylim(30, 75)
plt.legend(title="Age Group")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(years)
plt.show()
