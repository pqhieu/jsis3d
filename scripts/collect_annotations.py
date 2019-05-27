import os
import glob
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--root', help='path to root directory')
args = parser.parse_args()

root = args.root
fname = os.path.join(root, 'metadata', 'classes.txt')
classes = [line.strip() for line in open(fname, 'r')]
class2label = {cls: i for i, cls in enumerate(classes)}

fname = os.path.join(root, 'metadata', 'annotations.txt')
annotations = [line.strip() for line in open(fname, 'r')]
annotations = [os.path.join(root, 'raw', anno) for anno in annotations]

outdir = os.path.join(root, 'processed')
if not os.path.exists(outdir):
    os.mkdir(outdir)

def save_cloud_annotations(anno, fname):
    cloud = []
    flist = glob.glob(os.path.join(anno, '*.txt'))
    for i, fn in enumerate(flist):
        cls = os.path.basename(fn).split('_')[0]
        if cls not in classes:
            cls = 'clutter'
        pts = np.loadtxt(fn)
        sem = np.ones((pts.shape[0], 1)) * class2label[cls]
        ins = np.ones((pts.shape[0], 1)) * i
        pcd = np.concatenate([pts, sem, ins], axis=1)
        cloud.append(pcd)
    cloud  = np.concatenate(cloud, axis=0)
    origin = np.amin(cloud, axis=0)[0:3]
    cloud[:, 0:3] -= origin
    np.save(fname, cloud)

for anno in annotations:
    tokens = anno.split('/')
    fname = '{}_{}.npy'.format(tokens[-3], tokens[-2])
    fname = os.path.join(outdir, fname)
    print('> Writing point cloud to {}'.format(fname))
    if not os.path.exists(fname):
        save_cloud_annotations(anno, fname)
