from nilearn import image

# Load atlas and reference image
atlas_path = 'masks/desikanKillianyMNI.nii.gz'
ref_path = 'masks/example_cope.nii.gz'
atlas = image.load_img(atlas_path)
ref = image.load_img(ref_path)

# Resample atlas to match reference; use 'nearest' for labels
resampled_atlas = image.resample_to_img(atlas, ref, interpolation='nearest')
resampled_atlas.to_filename('masks/MVP_rois/desikanKillianyMNI-2mm.nii.gz')
