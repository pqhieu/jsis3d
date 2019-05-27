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

fname = os.path.join(root, 'metadata', 'train.txt')
flist = [os.path.join(root, 'h5', line.strip())
         for line in open(fname, 'r')]

fname = os.path.join(root, 'metadata', 'classes.txt')
classes = [line.strip() for line in open(fname, 'r')]
num_classes = len(classes)
sizes = [[] for i in range(num_classes)]

for fname in flist:
    print('> Processing {}...'.format(fname))
    fin = h5py.File(fname)
    coords = fin['coords'][:]
    points = fin['points'][:]
    labels = fin['labels'][:]
    labels = labels.reshape(-1, 2)

    for gid in np.unique(labels[:, 1]):
        indices = (labels[:, 1] == gid)
        cls = int(stats.mode(labels[indices, 0])[0])
        sizes[cls].append(np.sum(indices))

for i in range(num_classes):
    sizes[i] = np.mean(sizes[i]).astype(np.int32)

fname = os.path.join(root, 'metadata', 'sizes.txt')
print('> Saving statistics to {}...'.format(fname))
np.savetxt(fname, sizes, fmt='%d')
