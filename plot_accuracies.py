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
    predictions = []
    actuals = []
    code_list = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            seeds.append(int(row[0]))
            accuracies.append(float(row[1]))
            predictions.append(eval(row[2].replace('.', ',')))
            actuals.append(eval(row[3].replace('.', ',')))
            code_list.append(eval(row[4]))

    # Convert to numpy arrays for easier calculations
    seeds = np.array(seeds)
    accuracies = np.array(accuracies)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    code_list = np.array(code_list)

    return seeds, accuracies, predictions, actuals, code_list



if __name__ == '__main__':

    categories = []
    accuracies = np.array([])

    path = 'data/accuracies/cope_diff/'
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename) # Combine to get full path
        if os.path.isfile(full_path):
            print(filename)
            _, accuracies_reg, _, _, _ = get_accuracies(file_path=full_path)
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

        # Save to csv
        # Open the file in append mode ('a')
        with open('data/accuracies/mean_acc.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the new list of values as a single row
            writer.writerow([region_name, f"{mean:.4f}", f"{std:.4f}"])

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
