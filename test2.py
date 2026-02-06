from nilearn import plotting, datasets
from nilearn.surface import vol_to_surf
import nibabel as nib
import numpy as np

atlas_img = nib.load("masks/MVP_rois/desikanKillianyMNI-2mm.nii.gz")
atlas_data = atlas_img.get_fdata()

region_accuracy = {
    2: 0.7353,
    41: 0.7353,
    7: 0.7679,
    46: 0.7679,
    18: 0.7626,
    15: 0.7626,
    1009: 1.0,
    1008: 0.0,
    1015: 1.0
}

accuracy_map = np.zeros_like(atlas_data)

for region_id, acc in region_accuracy.items():
    accuracy_map[atlas_data == region_id] = acc

# Remove background zeros
#accuracy_map[atlas_data == 0] = np.nan

accuracy_img = nib.Nifti1Image(
    accuracy_map,
    affine=atlas_img.affine,
    header=atlas_img.header
)

nib.save(accuracy_img, "data/accuracies/maps/model_accuracy_map.nii.gz")

fsaverage = datasets.fetch_surf_fsaverage()

texture_left = vol_to_surf(
    accuracy_img,
    fsaverage.pial_left
)

texture_right = vol_to_surf(
    accuracy_img,
    fsaverage.pial_right
)


display = plotting.plot_surf_stat_map(
    fsaverage.infl_left,
    texture_left,
    hemi="left",
    cmap="viridis",
    colorbar=True,
    title="Model Accuracy – Left Hemisphere"
)

display.savefig("figures/model_accuracy_map.png", dpi=300)
plotting.show()