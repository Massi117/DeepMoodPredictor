# Plots the accuracies stored in data/accuracies.csv & calculates mean and std
import csv
import numpy as np
import matplotlib.pyplot as plt

# Read accuracies from CSV
seeds = []
accuracies = []
with open('data/accuracies.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        seeds.append(int(row[0]))
        accuracies.append(1 - float(row[1]))

# Convert to numpy arrays for easier calculations
seeds = np.array(seeds)
accuracies = np.array(accuracies)

# Calculate mean and standard deviation
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

# Print results
print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"Standard Deviation: {std_acc:.4f}")

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.plot(seeds, accuracies, marker='o', linestyle='', color='b', label='Accuracy per Seed')
plt.axhline(y=mean_acc, color='r', linestyle='--', label='Mean Accuracy')
plt.fill_between(seeds, mean_acc - std_acc, mean_acc + std_acc, color='r', alpha=0.2, label='±1 Std Dev')
plt.title('LOO Validation Accuracies')
plt.xlabel('Random Seed')
plt.ylabel('Accuracy')
plt.xticks(seeds)
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.savefig('figures/loo_accuracies.png')
plt.show()