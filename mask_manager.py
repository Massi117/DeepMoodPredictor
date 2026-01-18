# This script manages saliency maps and COPE data headers
import os
import numpy as np
import nibabel as nib

def get_header_info(nifti_path):
    """Retrieve header information from a NIfTI file.

    Args:
        nifti_path (str): Path to the NIfTI file.
    Returns:
        dict: Header information.
    """
    img = nib.load(nifti_path)
    header = img.header
    return dict(header)

if __name__ == '__main__':
    # Example usage
    nifti_file = 'masks/average_saliency_map_wholebrain.nii.gz'
    header_info = get_header_info(nifti_file)
    print("NIfTI Header Information:")
    for key, value in header_info.items():
        print(f"{key}: {value}")