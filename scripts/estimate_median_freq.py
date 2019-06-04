import os
import sys
import h5py
import argparse
import numpy as np
import scipy.stats as stats


parser = argparse.ArgumentParser()
parser.add_argument('--root', help='path to root directory')
args = parser.parse_args()

root = args.root

fname = os.path.join(root, 'metadata/train.txt')
flist = [os.path.join(root, 'h5', line.strip())
         for line in open(fname, 'r')]

fname = os.path.join(root, 'metadata', 'classes.txt')
classes = [line.strip() for line in open(fname, 'r')]
num_classes = len(classes)
sizes = np.zeros(num_classes)
total = np.zeros(num_classes)

for fname in flist:
    print('> Processing {}...'.format(fname))
    fin = h5py.File(fname)
    coords = fin['coords'][:]
    points = fin['points'][:]
    labels = fin['labels'][:]
    labels = labels.reshape(-1, 2)
    num_points = labels.shape[0]

    for i in range(num_classes):
        indices = (labels[:, 0] == i)
        size = np.sum(indices)
        sizes[i] += size
        if size == 0: continue
        total[i] += num_points

freq = sizes / total
weight = np.median(freq) / freq

fname = os.path.join(root, 'metadata', 'weight.txt')
print('> Saving statistics to {}...'.format(fname))
np.savetxt(fname, weight, fmt='%f')
