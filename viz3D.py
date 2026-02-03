# Plots the accuracies stored in data/accuracies.csv & calculates mean and std
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plot_accuracies as pa
import pyvista as pv
import nibabel as nib

if __name__ == '__main__':

    categories = []
    accuracies = np.array([])

    path = 'data/accuracies/cope1/'
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename) # Combine to get full path
        if os.path.isfile(full_path):
            print(filename)
            _, accuracies_reg = pa.get_accuracies(file_path=full_path)
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

    # 1. Load the .nii.gz file using nibabel
    nii_path = 'masks/MVP_rois/desikanKillianyMNI-2mm.nii.gz'
    img = nib.load(nii_path)
    data = img.get_fdata()

    # 2. Convert to a PyVista UniformGrid (ImageData)
    grid = pv.wrap(data)
    grid.spacing = img.header.get_zooms() # Set correct physical spacing

    # 3. Extract the surface (isosurface)
    # For atlas/labeled data, select a specific region (label) or 
    # threshold the data to find the brain boundary.
    # Example: threshold to find voxels > 0 (assuming 0 is background)
    surface = grid.contour([0.5], scalars=None) # Extracts isosurface

    # Alternatively, for label maps, extract a specific label (e.g., label 5)
    surface = grid.threshold([4.5, 5.5]).extract_surface()

    # 4. Smooth the surface for better visualization
    smoothed_surface = surface.smooth(n_iter=100)

    # Color the surface
    smoothed_surface = smoothed_surface.color_labels(colors='black')

    # 5. Plot the surface
    plotter = pv.Plotter()
    plotter.add_mesh(smoothed_surface, color='white', specular=0.5, smooth_shading=True)
    plotter.show()

    plotter.save_graphic('figures/brain_surface.png')
