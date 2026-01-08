# Package for data preprocessing and visualization

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

def nullify_background(data, bg_value=0):
    """Set background voxels to zero.

    Args:
        data (np.ndarray): 3D array of voxel data.
        bg_value (float): Value representing background.

    Returns:
        np.ndarray: Data with background voxels set to zero.
    """
    out = np.copy(data)
    out[out == bg_value] = None
    return out

def plot_cope_data(cope_data, save_path='figures/cope_sample.png'):
    """Plot COPE data vs voxel index.

    Args:
        cope_data (np.ndarray): 3D array of COPE values.
        save_path (str): Path to save the figure.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    flattened_cope = cope_data.flatten()
    inds = np.arange(flattened_cope.shape[0])

    plt.figure(figsize=(12, 4))
    plt.plot(inds, flattened_cope, lw=0.6, alpha=0.8, color='C0')
    plt.xlabel('Voxel index (flattened)')
    plt.ylabel('COPE value')
    plt.title('COPE values across voxels')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_cope_histogram(cope_data, save_path='figures/cope_histogram.png', bins=50):
    """Plot histogram of COPE values.

    Args:
        cope_data (np.ndarray): 3D array of COPE values.
        save_path (str): Path to save the figure.
        bins (int): Number of bins for the histogram.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    flattened_cope = cope_data.flatten()

    plt.figure(figsize=(8, 6))
    plt.hist(flattened_cope, bins=bins, color='C1', alpha=0.8, edgecolor='k')
    plt.xlabel('COPE value')
    plt.ylabel('Frequency')
    plt.title('Histogram of COPE values')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# Driver
if __name__ == '__main__':

    # Load data
    X = np.load('data/X_TRAIN_RAW.npy')
    y = np.load('data/y_TRAIN_RAW.npy')
    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")

    # Example usage
    subject = nullify_background(X[0])  # First subject
    label = y[0]
    print(f"Subject label: {label}")
    print(f'Subject mean COPE value (non-bg): {np.nanmean(subject)}')
    plot_cope_histogram(subject, save_path='figures/cope_histogram.png', bins=500)
    print("Saved COPE data plot to figures/cope_histogram.png")

    # Example usage
    subject = nullify_background(X[1])  # First subject
    label = y[1]
    print(f"Subject label: {label}")
    print(f'Subject mean COPE value (non-bg): {np.nanmean(subject)}')
    plot_cope_histogram(subject, save_path='figures/cope_histogram1.png', bins=500)
    print("Saved COPE data plot to figures/cope_histogram1.png")

    # Calculate the mean COPE value for responders vs non-responders
    responders = []
    non_responders = []
    for i in range(X.shape[0]):
        subject = nullify_background(X[i])
        subject[subject < 0] = None
        label = y[i]
        mean_cope = np.nanmean(subject)
        if label == 1:
            responders.append(mean_cope)
        else:
            non_responders.append(mean_cope)

    print(f'Mean COPE value for responders: {np.mean(responders)}')
    print(responders)
    print(f'Mean COPE value for non-responders: {np.mean(non_responders)}')
    print(non_responders)

    # Plot the means for non-responders vs responders with error bars
    plt.figure(figsize=(6, 4))
    plt.bar(['Non-responders', 'Responders'], [np.mean(non_responders), np.mean(responders)],
            yerr=[np.std(non_responders), np.std(responders)], color=['C0', 'C1'], alpha=0.8, edgecolor='k')
    plt.ylabel('Mean COPE value')
    plt.title('Mean COPE values: Non-responders vs Responders')
    plt.tight_layout()
    plt.savefig('figures/mean_cope_comparison.png', dpi=150)
    plt.close()
    print("Saved mean COPE comparison plot to figures/mean_cope_comparison.png")