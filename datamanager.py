import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

def load_data(cope_type='cope_diff', continuous_labels=False, mask_dir=None):
    """Load COPE data for responders and non-responders.
    Args:
        cope_type (str): Type of COPE data to load ('cope_diff' or other).
        continuous_labels (bool): Whether to use continuous MDD change scores as labels.
    Returns:
        X (np.ndarray): Voxelwise COPE data.
        y (np.ndarray): Labels and MDD change scores."""

    # === Paths & Data ===
    feat_base = '/home/mbosli/DeepMoodPredictor/data/cope_data/forMassi/mumfordadvice'
    subj_lists = {
        'nonresponder': '/home/mbosli/DeepMoodPredictor/data/grps/nonresponder.txt',
        'responder': '/home/mbosli/DeepMoodPredictor/data/grps/responder.txt'
    }
    mdd_codes = [11, 13, 17, 18, 21, 24, 31, 29, 32, 35, 37, 28, 43, 60, 46, 51, 61, 63, 62, 65, 
            66, 69, 70, 71, 72, 73, 78, 75, 76, 79, 80, 81, 82, 83, 84, 85, 86, 87, 90, 92, 93, 94]
    #mdd_codes = [13, 17, 21, 32, 46, 51, 62, 63, 69, 71, 72, 73, 79, 80, 81, 83, 85, 86, 90, 92, 93]
    mdd_wr = [22, 19, 20, 18, 21, 14, 19, 20, 28, 26, 28, 20, 23, 18, 24, 27, 15, 20, 19, 21, 25, 
            20, 22, 25, 17, 18, 28, 21, 27, 22, 17, 19, 23, 20, 31, 22, 16, 29, 20, 20, 29, 26]
    mdd_sd = [20, 4, 5, 16, 26, 13, 14, 11, 9, 21, 16, 9, 21, 10, 3, 28, 3, 20, 2, 9, 19, 23, 20,
            11, 22, 4, 26, 8, 20, 23, 29, 3, 19, 21, 27, 6, 17, 26, 6, 24, 11, 19]
    mdd_change = [sd - wr for sd, wr in zip(mdd_sd, mdd_wr)]

    # Check if a mask is provided
    if mask_dir is None:
        use_mask = False
    else:
        use_mask = True
        mask_data = nib.load(mask_dir).get_fdata()
        mask_indices = np.where(mask_data > 0)

    # === Load subjects ===
    def load_subjects(path):
        with open(path, 'r') as f:
            return [int(s) for s in f.read().strip().split()]

    subjects = {}
    for group, path in subj_lists.items():
        subjects[group] = load_subjects(path)

    # === Extract voxelwise COPE values ===
    X, y = [], []

    for group, codes in subjects.items():
        label = 1 if group == 'responder' else 0
        for code in codes:
            print(f"Loading subject m{code} from group {group}...")
            cope_path = os.path.join(
                feat_base, group, f"m{code}.gfeat", "cope1.feat", "stats", f"{cope_type}.nii.gz"
            )
            if not os.path.exists(cope_path):
                print(f"Missing: {cope_path}")
                continue
            try:
                cope_data = nib.load(cope_path).get_fdata()
                voxels = cope_data
                if use_mask:
                    voxels[mask_indices] = 0
                idx = mdd_codes.index(code)
                X.append(voxels)
                mdd_change = mdd_wr[idx] - mdd_sd[idx]
                y.append([label, mdd_change])
            except Exception as e:
                print(f"Error loading {cope_path}: {e}")

    # Transform to np arrays
    X = np.array(X)
    y = np.array(y)

    # Split y into classification and regression parts
    y_class = np.array([])
    y_r = np.array([])
    for sublist in y:
        y_class = np.append(y_class, sublist[0])
        y_r = np.append(y_r, sublist[1])

    if continuous_labels:
        y = y_r
    else:
        y = y_class

    # Save raw data
    np.save('data/X_RAW.npy', X)
    np.save('data/y_RAW.npy', y)

    print(f"Loaded {len(X)} binary samples (NR vs R). Feature shape: {X.shape}")

    return X, y

def create_train_test_split(X, y, test_size=0.1, random_state=None):
    """Split the data into training and testing sets.
    Args:
        X (np.ndarray): Voxelwise COPE data.
        y (np.ndarray): Labels and MDD change scores.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test: Split datasets."""

    # Make the split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )


    # Save split raw data
    np.save('data/X_TRAIN_RAW.npy', X_train)
    np.save('data/y_TRAIN_RAW.npy', y_train)
    np.save('data/X_TEST_RAW.npy', X_test)
    np.save('data/y_TEST_RAW.npy', y_test)

    #np.save('data/y_TEST_RAW_REG.npy', y_test_r)
    #np.save('data/y_TRAIN_RAW_REG.npy', y_train_r)


# Driver
if __name__ == '__main__':

    # Load the data
    X, y = load_data(cope_type='cope_diff')

    # Create train-test split
    print("Creating train-test split...")
    create_train_test_split(X, y, test_size=0.05)
    print("Done.")