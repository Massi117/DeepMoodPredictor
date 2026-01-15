import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

def get_data_parsed(mask_path=None):
    """
    Loads specified cope data while trimming the resulting array to the mask size
    """

    # === Paths ===
    if mask_path != None:
        use_mask = True
    else:
        use_mask = False
    feat_base = '/home/mbosli/DeepMoodPredictor/data/cope_data/forMassi/mumfordadvice'
    subj_lists = {
        'nonresponder': '/home/mbosli/DeepMoodPredictor/data/grps/nonresponder.txt',
        'responder': '/home/mbosli/DeepMoodPredictor/data/grps/responder.txt'
    }

    # === Load V1 mask ===
    if use_mask:
        mask_data = nib.load(mask_path).get_fdata()
        mask_indices = np.where(mask_data > 0)

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

    for group, codes in subjects.items():
        label = 1 if group == 'responder' else 0
        for code in codes:
            cope_path = os.path.join(
                feat_base, group, f"m{code}.gfeat", "cope1.feat", "stats", "cope_diff.nii.gz"
            )
            if not os.path.exists(cope_path):
                print(f"Missing: {cope_path}")
                continue
            try:
                cope_data = nib.load(cope_path).get_fdata()
                if use_mask:
                    filter_mat = np.zeros_like(cope_data)
                    filter_mat[mask_indices] = 1
                    voxels = trim3D(cope_data, filter_mat, trim_value=0)

                else:
                    voxels = cope_data
                idx = mdd_codes.index(code)
                X.append(voxels)
                mdd_change = mdd_wr[idx] - mdd_sd[idx]
                y.append([label, mdd_change])
            except Exception as e:
                print(f"Error loading {cope_path}: {e}")

    # Transform to np arrays
    X = np.array(X)
    y = np.array(y)

    # Flatten X values but preserve locations
    if use_mask == False:
        flattened = X.reshape(X.shape[0], -1)  # Shape: (39, 91*109*91)
    else:
        flattened = np.array([])

    # Generate the index mapping
    # First, create all combinations of indices for the last 3 dims
    sub_indices = np.indices((91, 109, 91)).reshape(3, -1).T  # Shape: (91*109*91, 3)

    # Now tile this across the first dimension (39)
    num_per_outer = sub_indices.shape[0]  # 91*109*91
    outer_indices = np.repeat(np.arange(39), num_per_outer).reshape(-1, 1)  # Shape: (39*num_per_outer, 1)

    # Repeat sub_indices for each outer index
    all_indices = np.hstack([outer_indices, np.tile(sub_indices, (39, 1))])  # Shape: (39*num_per_outer, 4)
    all_indices = np.delete(all_indices, 0, axis=1)

    print(f"Loaded {len(X)} binary samples (NR vs R). Feature shape: {X.shape}")

    return X, flattened, y, all_indices


def trim3D(filt, kernel, trim_value=0):
    """
    Trims a 3D array based on the kernel
    """

    (x, y, z) = filt.shape

    # Trim axis x
    x_start = 0
    for i in range(x):
        if (kernel[i,:,:] == trim_value).all():
            x_start = x_start + 1
        else:
            break

    x_end = x
    for i in range(x):
        if (kernel[x-1-i] == trim_value).all():
            x_end = x_end - 1
        else:
            break

    # Trim axis y
    y_start = 0
    for i in range(y):
        if (kernel[:,i] == trim_value).all():
            y_start = y_start + 1
        else:
            break

    y_end = y
    for i in range(y):
        if (kernel[:,y-1-i] == trim_value).all():
            y_end = y_end - 1
        else:
            break

    # Trim axis z
    z_start = 0
    for i in range(z):
        if (kernel[:,:,i] == trim_value).all():
            z_start = z_start + 1
        else:
            break

    z_end = z
    for i in range(z):
        if (kernel[:,:,z-1-i] == trim_value).all():
            z_end = z_end - 1
        else:
            break

    filtered = filt[x_start:x_end, y_start:y_end, z_start:z_end]

    return filtered



# Driver
if __name__ == '__main__':

    # Load the data
    print('Loading data...')
    X, X_flat, y, locs = get_data_parsed('/home/mbosli/DeepMoodPredictor/masks/MVP_rois/Thalamus_mask.nii.gz')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

    # Split the y_data after split
    print('Splitting the data...')
    y_train_c = []
    y_train_r = []

    for sublist in y_train:
        y_train_c.append(sublist[0])
        y_train_r.append(sublist[1])

    y_test_c = []
    y_test_r = []

    for sublist in y_test:
        y_test_c.append(sublist[0])
        y_test_r.append(sublist[1])


    # Save split raw data
    np.save('data/X_TRAIN_RAW.npy', X_train)
    np.save('data/y_TRAIN_RAW.npy', y_train_c)
    np.save('data/X_TEST_RAW.npy', X_test)
    np.save('data/y_TEST_RAW.npy', y_test_c)

    np.save('data/y_TEST_RAW_REG.npy', y_test_r)
    np.save('data/y_TRAIN_RAW_REG.npy', y_train_r)

    # Save raw data
    np.save('data/X_RAW.npy', X)
    np.save('data/y_RAW.npy', y) 
