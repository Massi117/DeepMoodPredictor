#!/usr/bin/env python3
"""Compute ANOVA F-values between classes for train and test sets and plot results.

Usage:
    python scripts/plot_fvalues.py

This script expects the following files (defaults):
    data/X_TRAIN_RAW.npy, data/y_TRAIN_RAW.npy
    data/X_TEST_RAW.npy,  data/y_TEST_RAW.npy

Outputs:
    figures/fvalues_hist.png
    figures/top20_fvalues.png
    data/fvalues_train.npy
    data/fvalues_test.npy
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif


def load_and_flat(path):
    arr = np.load(path)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def compute_fvalues(X, y):
    # scikit-learn expects finite numeric arrays; ensure dtype=float
    Xf = np.asarray(X, dtype=float)
    yv = np.asarray(y).ravel()
    fvals, pvals = f_classif(Xf, yv)
    return fvals, pvals


def plot_hist_compare(f_train, f_test, out_path='figures/fvalues_hist.png'):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(f_train, bins=120, alpha=0.6, label='Train', density=False)
    plt.hist(f_test, bins=120, alpha=0.6, label='Test', density=False)
    plt.xlabel('F-value')
    plt.ylabel('Feature count')
    plt.legend()
    plt.title('Distribution of ANOVA F-values (features)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_top_features(f_train, top_n=20, out_path='figures/top20_fvalues.png'):
    """Plot F-statistic for each voxel (x = voxel index, y = F-value).

    Highlights the top `top_n` features on the same plot.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    f_train = np.asarray(f_train)
    n_feats = f_train.shape[0]

    # indices and top indices
    inds = np.arange(n_feats)
    top_idx = np.argsort(f_train)[::-1][:top_n]

    plt.figure(figsize=(12, 4))
    plt.plot(inds, f_train, lw=0.6, alpha=0.8, color='C0')
    # highlight top features
    plt.scatter(top_idx, f_train[top_idx], color='C1', edgecolor='k', zorder=5, label=f'Top {top_n}')

    plt.xlabel('Voxel index (flattened)')
    plt.ylabel('F-value')
    plt.title('Feature F-statistic across voxels (train)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_fvalues_vs_index(f_train, f_test, out_path='figures/fvalues_vs_index.png'):
    """Plot all F-statistics for train and test sets vs voxel index on the same chart."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    f_train = np.asarray(f_train)
    f_test = np.asarray(f_test)

    n = min(f_train.shape[0], f_test.shape[0])
    inds = np.arange(n)

    plt.figure(figsize=(12, 4))
    plt.plot(inds, f_train[:n], lw=0.6, alpha=0.8, color='C0', label='Train')
    plt.plot(inds, f_test[:n], lw=0.6, alpha=0.8, color='C1', label='Test')
    plt.xlabel('Voxel index (flattened)')
    plt.ylabel('F-value')
    plt.title('ANOVA F-statistic across voxels (Train vs Test)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    X_train = load_and_flat(args.x_train)
    y_train = np.load(args.y_train)
    X_test = load_and_flat(args.x_test)
    y_test = np.load(args.y_test)

    print('Shapes:')
    print('  X_train', X_train.shape)
    print('  y_train', y_train.shape)
    print('  X_test ', X_test.shape)
    print('  y_test ', y_test.shape)

    f_train, p_train = compute_fvalues(X_train, y_train)
    f_test, p_test = compute_fvalues(X_test, y_test)

    os.makedirs('data', exist_ok=True)
    np.save('data/fvalues_train.npy', f_train)
    np.save('data/fvalues_test.npy', f_test)

    plot_hist_compare(f_train, f_test, out_path=os.path.join('figures', 'fvalues_hist.png'))
    plot_fvalues_vs_index(f_train, f_test, out_path=os.path.join('figures', 'fvalues_vs_index.png'))

    print('Saved f-values arrays to data/fvalues_train.npy and data/fvalues_test.npy')
    print('Saved figures to figures/fvalues_hist.png and figures/fvalues_vs_index.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Compute and plot ANOVA F-values between classes')
    p.add_argument('--x-train', dest='x_train', default='data/X_TRAIN_RAW.npy')
    p.add_argument('--y-train', dest='y_train', default='data/y_TRAIN_RAW.npy')
    p.add_argument('--x-test', dest='x_test', default='data/X_TEST_RAW.npy')
    p.add_argument('--y-test', dest='y_test', default='data/y_TEST_RAW.npy')
    p.add_argument('--top-n', dest='top_n', type=int, default=20)
    args = p.parse_args()
    main(args)
