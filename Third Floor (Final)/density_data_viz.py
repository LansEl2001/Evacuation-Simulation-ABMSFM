import pandas as pd
import matplotlib.pyplot as plt

file_path = 'density_data.txt'
data = pd.read_csv(file_path, header=None, delim_whitespace=True)

#Adjust Data Points
data.iloc[:, -3:] = data.iloc[:, -3:] * 1e6
data_sorted = data.sort_values(by=0) #sort
data_clean_sorted = data_sorted.drop_duplicates(subset=[0], keep='first') #remove duplicates

# rolling mean
data_smoothed = data_clean_sorted.rolling(window=500, min_periods=1).mean()


colors = ['blue', 'green', 'red']
exit_labels = ['Bottom Left Stairs', 'Bottom Right Stairs', 'Fire Exit (3F)']

plt.figure(figsize=(12, 8))
for i, (color, label) in enumerate(zip(colors, exit_labels), start=1):
    plt.plot(data_smoothed.iloc[:, 0], data_smoothed.iloc[:, -i], label=label, color=color, linewidth=2)

plt.xlabel('Time')
plt.ylabel('Density')
plt.title('Density Over Time for Each Exit (3rd Floor)')
plt.legend()
plt.grid(True)
plt.show()