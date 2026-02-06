import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from plot_accuracies import get_accuracies

# Load the data
full_path = 'data/accuracies/cope_diff/accuracies_amygdala.csv'
seeds, accs, preds, labels, codes = get_accuracies(file_path=full_path)
print(f'Mean accuracy: {np.mean(accs)}')


# Perform One-Sample T-Test
t_stat, p_val = ttest_1samp(accs, 0.5)

# Output results
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val}")

# 4. Interpretation
alpha = 0.05
if p_val < alpha:
    print("Reject null hypothesis (significant difference)")
else:
    print("Fail to reject null hypothesis (no significant difference)")



