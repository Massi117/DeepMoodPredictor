import nibabel as nib
import numpy as np
import csv

def extract_label_as_mask(input_path, target_label, output_path):
    """
    Extracts a specific label from a multi-label NIfTI file 
    and saves it as a binary mask NIfTI file.
    Parameters:
    - input_path: str, path to the input NIfTI file with multiple labels
    - target_label: list of int, label(s) to extract
    - output_path: str, path to save the output binary mask NIfTI file
    """
    # Load the image
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine

    # Create the binary mask
    final_mask = np.zeros(data.shape, dtype=np.uint8)   
    for label in target_label:
        #print(f"Extracting label {label}...")
        binary_mask = (data == label)
        binary_mask_int = binary_mask.astype(np.uint8) # Convert to 0/1 integers
        final_mask = np.maximum(final_mask, binary_mask_int)

    # Create a new Nifti image and save
    mask_img = nib.Nifti1Image(final_mask, affine)
    nib.save(mask_img, output_path)
    print(f"Successfully saved binary mask for labels {target_label} to {output_path}")

# Usage
if __name__ == "__main__":
    input_nifti_path = 'masks/MVP_rois/desikanKillianyMNI-2mm.nii.gz'
    idx_path = 'masks/MVP_rois/atlas_index.csv'
    with open(idx_path, 'r') as f:
        reader = csv.reader(f)        
        # Transpose rows to columns
        columns = list(zip(*reader))

    # Now each column is a tuple (easily converted to a list)
    names = list(columns[0])
    idx = list(columns[1])
    idx = [int(i) for i in idx]

    filename = 'thalamus'
    regions_of_interest = [
        'Right-Thalamus-Proper',
        'Left-Thalamus-Proper'
    ]


    label_to_extract = idx
    for region in regions_of_interest:
        print(len(idx))
        if region in names:
            index_position = names.index(region)
            label = int(idx[index_position])
            print(f"Found region {region} at index {index_position} with label {label}")
            label_to_extract = [item for item in label_to_extract if item != label]
        else:
            print(f"Region {region} not found in index file.")

    output_mask_path = f'masks/MVP_rois/{filename}-mask.nii.gz'
    extract_label_as_mask(input_nifti_path, label_to_extract, output_mask_path)
        