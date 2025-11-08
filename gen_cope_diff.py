import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

def get_cope_diff():

    # === Paths ===
    feat_base = '/home/mbosli/DeepMoodPredictor/data/cope_data/forMassi/mumfordadvice'
    subj_lists = {
        'nonresponder': '/home/mbosli/DeepMoodPredictor/data/grps/nonresponder.txt',
        'responder': '/home/mbosli/DeepMoodPredictor/data/grps/responder.txt'
    }

    # === Load subjects ===
    def load_subjects(path):
        with open(path, 'r') as f:
            return [int(s) for s in f.read().strip().split()]

    subjects = {}
    for group, path in subj_lists.items():
        subjects[group] = load_subjects(path)

    # === Extract voxelwise COPE2 values ===
    X, y = [], []

    # Codes and scores
    mdd_codes = [11, 13, 17, 18, 21, 24, 31, 29, 32, 35, 37, 28, 43, 60, 46, 51, 61, 63, 62, 65, 
            66, 69, 70, 71, 72, 73, 78, 75, 76, 79, 80, 81, 82, 83, 84, 85, 86, 87, 90, 92, 93, 94]
    mdd_wr = [22, 19, 20, 18, 21, 14, 19, 20, 28, 26, 28, 20, 23, 18, 24, 27, 15, 20, 19, 21, 25, 
            20, 22, 25, 17, 18, 28, 21, 27, 22, 17, 19, 23, 20, 31, 22, 16, 29, 20, 20, 29, 26]
    mdd_sd = [20, 4, 5, 16, 26, 13, 14, 11, 9, 21, 16, 9, 21, 10, 3, 28, 3, 20, 2, 9, 19, 23, 20,
            11, 22, 4, 26, 8, 20, 23, 29, 3, 19, 21, 27, 6, 17, 26, 6, 24, 11, 19]
    
    ref_file = 'masks/MVP_rois/amygdalawhole_thr50_2mm.nii.gz'
    ref_img = nib.load(ref_file)
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine

    for group, codes in subjects.items():
        label = 1 if group == 'responder' else 0
        for code in codes:
            cope1_path = os.path.join(
                feat_base, group, f"m{code}.gfeat", "cope1.feat", "stats", "cope1.nii.gz"
            )
            cope2_path = os.path.join(
                feat_base, group, f"m{code}.gfeat", "cope1.feat", "stats", "cope2.nii.gz"
            )
            if not os.path.exists(cope1_path):
                print(f"Missing: {cope1_path}")
                continue
            if not os.path.exists(cope2_path):
                print(f"Missing: {cope2_path}")
                continue
            try:
                cope1_data = nib.load(cope1_path).get_fdata()
                cope2_data = nib.load(cope2_path).get_fdata()

                cope_diff = cope2_data - cope1_data

                # Create a new NIfTI image
                diff_data = nib.Nifti1Image(cope_diff, affine=ref_affine)

                # Save to disk
                print('Saving mask to disk....')
                save_path = os.path.join(
                    feat_base, group, f"m{code}.gfeat", "cope1.feat", "stats", "cope_diff.nii.gz"
                )
                nib.save(diff_data, save_path)
                print('DONE')
                

            except Exception as e:
                print(f"Error loading {cope1_path}: {e}")

    print(f"Completed cope data generation.")



# Driver
if __name__ == '__main__':

    # Load the data
    get_cope_diff()