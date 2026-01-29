# Plots the accuracies stored in data/accuracies.csv & calculates mean and std
import csv
import numpy as np
import pandas as pd
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
    # Load data
    whole_path = 'data/accuracies_.csv'
    amygdala_path = 'data/accuracies_amygdala.csv'
    thal_path = 'data/archive/accuracies_thalamus.csv'
    _, accuracies_whole = get_accuracies(file_path=whole_path)
    _, accuracies_amyg = get_accuracies(file_path=amygdala_path)
    _, accuracies_thal = get_accuracies(file_path=thal_path)

    # Calculate mean and standard deviation
    mean_whole = np.mean(accuracies_whole)
    std_whole = np.std(accuracies_whole)

    mean_amyg = np.mean(accuracies_amyg)
    std_amyg = np.std(accuracies_amyg)

    mean_thal = np.mean(accuracies_thal)
    std_thal = np.std(accuracies_thal)

    # Print results
    print(f"Mean Accuracy (whole brain): {mean_whole:.4f}")
    print(f"Standard Deviation (whole brain): {std_whole:.4f}")

    print(f"Mean Accuracy (amygdala): {mean_amyg:.4f}")
    print(f"Standard Deviation (amygdala): {std_amyg:.4f}")

    print(f"Mean Accuracy (thalamus): {mean_thal:.4f}")
    print(f"Standard Deviation (thalamus): {std_thal:.4f}")

    categories = (['Whole Brain'] * len(accuracies_whole)) + (['Amygdala'] * len(accuracies_amyg))
    accuracies = np.concatenate((accuracies_whole, accuracies_amyg))

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
    '''
        showcaps=False,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':0},
        hue="alive", fill=False, gap=.1)
    '''
    plt.savefig('figures/loo_accuracies_test.png')

    '''
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, marker='o', linestyle='', color='b', label='Accuracy per Seed')
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
    '''
