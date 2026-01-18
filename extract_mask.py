import nibabel as nib
import numpy as np
import os

def extract_label_as_mask(input_path, target_label, output_path):
    """
    Extracts a specific label from a multi-label NIfTI file 
    and saves it as a binary mask NIfTI file.
    """
    # Load the image
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine

    # Create the binary mask
    final_mask = np.zeros(data.shape, dtype=np.uint8)   
    for label in target_label:
        print(f"Extracting label {label}...")
        binary_mask = (data == label)
        binary_mask_int = binary_mask.astype(np.uint8) # Convert to 0/1 integers
        final_mask = np.maximum(final_mask, binary_mask_int)

    # Create a new Nifti image and save
    mask_img = nib.Nifti1Image(final_mask, affine)
    nib.save(mask_img, output_path)
    print(f"Successfully saved binary mask for labels {target_label} to {output_path}")

# Usage
if __name__ == "__main__":
    input_nifti_path = 'masks/MVP_rois/HarvardOxford-sub-maxprob-thr50-2mm.nii.gz'
    label_to_extract = [0,
                        1, 
                        2, 
                        3, 
                        4, 
                        5, 
                        6, 
                        7, 
                        8, 
                        #9, 
                        10,
                        11,
                        12,
                        13, 
                        14, 
                        15, 
                        16, 
                        17, 
                        18, 
                        #19, 
                        20, 
                        21]  # Example label
    output_mask_path = 'masks/MVP_rois/hippocampus-thr50-2mm.nii.gz'
    
    extract_label_as_mask(input_nifti_path, label_to_extract, output_mask_path)