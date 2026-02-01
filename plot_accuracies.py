# Plots the accuracies stored in data/accuracies.csv & calculates mean and std
import csv
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def get_accuracies(file_path):
    # Read accuracies from CSV
    seeds = []
    accuracies = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            seeds.append(int(row[0]))
            accuracies.append(float(row[1]))

    # Convert to numpy arrays for easier calculations
    seeds = np.array(seeds)
    accuracies = np.array(accuracies)

    return seeds, accuracies



if __name__ == '__main__':

    categories = []
    accuracies = np.array([])

    path = 'data/accuracies/cope1/'
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename) # Combine to get full path
        if os.path.isfile(full_path):
            print(filename)
            _, accuracies_reg = get_accuracies(file_path=full_path)
        else:
            print(f'{filename} is not a file.')
            continue
        # Calculate mean and standard deviation
        mean = np.mean(accuracies_reg)
        std = np.std(accuracies_reg)
        
        # Print results
        region_name = filename.replace('accuracies_', '').replace('.csv', '')
        print(f"Mean Accuracy ({region_name}): {mean:.4f}")
        print(f"Standard Deviation ({region_name}): {std:.4f}")

        # Add data to list
        categories = categories + ([region_name] * len(accuracies_reg))
        accuracies = np.concatenate((accuracies, accuracies_reg))

    # Create a DataFrame
    data = {
        'Brain Region': categories,
        'Accuracy': accuracies
    }
    df = pd.DataFrame(data)

    # Plot
    sns.set_style("whitegrid")
    ax = sns.swarmplot(x="Brain Region", y="Accuracy", data=df)
    ax = sns.boxplot(x="Brain Region", y="Accuracy", data=df,
                     flierprops={"marker": "x"},
                    boxprops={"facecolor": (.3, .5, .7, .5)},
                    medianprops={"color": "r", "linewidth": 2},)

    plt.savefig('figures/loo_accuracies_test.png')
