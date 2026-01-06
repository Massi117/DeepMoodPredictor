import os
import numpy as np
import matplotlib.pyplot as plt


# Codes and scores (module scope so other functions can use them)
mdd_codes = [11, 13, 17, 18, 21, 24, 31, 29, 32, 35, 37, 28, 43, 60, 46, 51, 61, 63, 62, 65,
             66, 69, 70, 71, 72, 73, 78, 75, 76, 79, 80, 81, 82, 83, 84, 85, 86, 87, 90, 92, 93, 94]
mdd_wr = [22, 19, 20, 18, 21, 14, 19, 20, 28, 26, 28, 20, 23, 18, 24, 27, 15, 20, 19, 21, 25,
          20, 22, 25, 17, 18, 28, 21, 27, 22, 17, 19, 23, 20, 31, 22, 16, 29, 20, 20, 29, 26]
mdd_sd = [20, 4, 5, 16, 26, 13, 14, 11, 9, 21, 16, 9, 21, 10, 3, 28, 3, 20, 2, 9, 19, 23, 20,
          11, 22, 4, 26, 8, 20, 23, 29, 3, 19, 21, 27, 6, 17, 26, 6, 24, 11, 19]


def compute_mdd_change(codes=None, wr=None, sd=None):
    """Compute per-subject MDD change = wr - sd.

    Returns:
        mapping (dict): code -> change
        codes_arr (np.ndarray): codes in input order
        changes_arr (np.ndarray): corresponding changes in same order
    """
    codes = mdd_codes if codes is None else codes
    wr = mdd_wr if wr is None else wr
    sd = mdd_sd if sd is None else sd

    if not (len(codes) == len(wr) == len(sd)):
        raise ValueError("`codes`, `wr`, and `sd` must have the same length")

    changes = [int(w) - int(s) for w, s in zip(wr, sd)]
    mapping = dict(zip(codes, changes))
    return mapping, np.array(codes, dtype=int), np.array(changes, dtype=int)


def load_group_codes(grp_dir='data/grps'):
    """Load responder and nonresponder code lists from `data/grps`.

    Returns:
        dict: {'responder': [codes], 'nonresponder': [codes]}
    """
    groups = {'responder': os.path.join(grp_dir, 'responder.txt'),
              'nonresponder': os.path.join(grp_dir, 'nonresponder.txt')}
    out = {}
    for k, p in groups.items():
        try:
            with open(p, 'r') as f:
                # assume space/newline separated integers
                out[k] = [int(s) for s in f.read().strip().split() if s.strip()]
        except FileNotFoundError:
            out[k] = []
    return out


def plot_group_changes(mapping, groups=None, save_path='figures/group_changes.png'):
    """Plot responders vs nonresponders using the mapping code->change.

    Produces a boxplot with overlaid scatter of individual subject changes.
    """
    if groups is None:
        groups = load_group_codes()

    resp_codes = groups.get('responder', [])
    nonresp_codes = groups.get('nonresponder', [])

    resp_changes = [mapping[c] for c in resp_codes if c in mapping]
    nonresp_changes = [mapping[c] for c in nonresp_codes if c in mapping]

    figdir = os.path.dirname(save_path)
    if figdir:
        os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([nonresp_changes, resp_changes], tick_labels=['Nonresponders', 'Responders'], patch_artist=True)
    # overlay jittered scatter
    for i, vals in enumerate([nonresp_changes, resp_changes], start=1):
        x = np.random.normal(i, 0.06, size=len(vals))
        ax.scatter(x, vals, alpha=0.8, edgecolor='k')

    ax.set_ylabel('MDD change (wr - sd)')
    ax.set_title('Responders vs Nonresponders — MDD Change')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def find_codes_diff_from_value(mapping=None, target=7, min_diff=3):
    """Return sorted list of subject codes whose change differs from `target` by at least `min_diff`.

    Args:
        mapping (dict): code -> change mapping. If None, will call `compute_mdd_change()`.
        target (int): target value to compare against (default 7).
        min_diff (int): minimum absolute difference required (default 3).

    Returns:
        list[int]: sorted subject codes meeting the criterion.
    """
    if mapping is None:
        mapping, _, _ = compute_mdd_change()
    out = [int(code) for code, ch in mapping.items() if abs(ch - target) >= min_diff]
    return sorted(out)


if __name__ == '__main__':
    # Compute mapping and save arrays
    mapping, codes_arr, changes_arr = compute_mdd_change()
    out_dir = 'data'
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'mdd_codes.npy'), codes_arr)
    np.save(os.path.join(out_dir, 'mdd_changes.npy'), changes_arr)

    # Print compact table
    for c, ch in zip(codes_arr, changes_arr):
        print(f"Code {c}: change {ch}")

    # Plot groups
    groups = load_group_codes()
    fig_path = plot_group_changes(mapping, groups=groups, save_path='figures/group_changes.png')
    print(f"Saved plot to {fig_path}")

    # Find and save codes with change at least 3 away from 7
    codes_diff = find_codes_diff_from_value(mapping=mapping, target=7, min_diff=7)
    print("Codes with |(wr-sd) - 7| >= 7:", codes_diff)
    np.save(os.path.join(out_dir, 'mdd_codes_diff_from_7.npy'), np.array(codes_diff, dtype=int))
    
	