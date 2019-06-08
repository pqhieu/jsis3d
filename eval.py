import os
import json
import h5py
import argparse
import numpy as np
import plyfile as ply
import scipy.stats as stats

from utils import block_merge


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='path to the logging directory')
parser.add_argument('--visualize', action='store_true', help='save point clouds to PLY files')
args = parser.parse_args()

logdir = args.logdir
visualize = args.visualize
config = os.path.join(logdir, 'config.json')
args = json.load(open(config))

fname = os.path.join(logdir, 'pred.npz')
print('> Loading predictions from {}...'.format(fname))
pdict = np.load(fname)
pdict = np.stack([pdict['semantics'], pdict['instances']], axis=-1)

fname = os.path.join(args['root'], 'metadata', 'classes.txt')
classes = [line.strip() for line in open(fname, 'r')]
fname = os.path.join(args['root'], 'metadata', 'sizes.txt')
sizes = np.loadtxt(fname)
fname = os.path.join(args['root'], 'metadata', 'test.txt')
flist = [line.strip() for line in open(fname)]

num_classes = args['num_classes']
accu  = np.zeros(num_classes)
freq  = np.zeros(num_classes)
inter = np.zeros(num_classes)
union = np.zeros(num_classes)
total = np.zeros(num_classes)
fps   = [[] for i in range(num_classes)]
tps   = [[] for i in range(num_classes)]

offset = 0
for fname in flist:
    print('> Evaluating on {}...'.format(fname))
    fname = os.path.join(args['root'], 'h5', fname)
    fin = h5py.File(fname)
    coords = fin['coords'][:]
    points = fin['points'][:]
    labels = fin['labels'][:]
    step   = coords.shape[0]

    pred = pdict[offset:offset+step]
    pred = pred.reshape(-1, 2)

    coords = coords.reshape(-1, 3)
    points = points.reshape(-1, 9)
    truth  = labels.reshape(-1, 2)
    num_points = coords.shape[0]

    # evaluate semantic accuracy & IoU
    for i in range(num_classes):
        indices = (truth[:, 0] == i)
        correct = (pred[indices, 0] == truth[indices, 0])
        accu[i]  += np.sum(correct)
        freq[i]  += np.sum(indices)
        inter[i] += np.sum((pred[:, 0] == i) & (truth[:, 0] == i))
        union[i] += np.sum((pred[:, 0] == i) | (truth[:, 0] == i))

    # evaluate instance mAP
    proposals = [[] for i in range(num_classes)]
    for gid in np.unique(pred[:, 1]):
        indices = (pred[:, 1] == gid)
        cls = int(stats.mode(pred[indices, 0])[0])
        size = np.sum(indices)
        if size > 0.25 * sizes[cls]: # remove small instances
            proposals[cls] += [indices]

    instances = [[] for i in range(num_classes)]
    for gid in np.unique(truth[:, 1]):
        indices = (truth[:, 1] == gid)
        cls = int(stats.mode(truth[indices, 0])[0])
        instances[cls] += [indices]

    for i in range(num_classes):
        total[i] += len(instances[i])
        tp = np.zeros(len(proposals[i]))
        fp = np.zeros(len(proposals[i]))
        gt = np.zeros(len(instances[i]))
        for pid, u in enumerate(proposals[i]):
            overlap = 0.0
            detected = 0
            for iid, v in enumerate(instances[i]):
                iou = np.sum((u & v)) / np.sum((u | v))
                if iou > overlap:
                    overlap = iou
                    detected = iid
            if overlap >= 0.5:
                tp[pid] = 1
            else:
                fp[pid] = 1
        tps[i] += [tp]
        fps[i] += [fp]

    if visualize:
        colors = (points[:, 3:6] * 255).astype(np.uint8)
        vertex = [(coords[i, 0], coords[i, 1], coords[i, 2],
                   colors[i, 0], colors[i, 1], colors[i, 2],
                   pred[i, 0], pred[i, 1] + 1)
                  for i in range(num_points)]
        vertex = np.array(vertex, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('nyu_class', 'u2'), ('label', 'u4')
        ])
        el = ply.PlyElement.describe(vertex, 'vertex')
        data = ply.PlyData([el], text=False)
        basename = os.path.splitext(os.path.basename(fname))[0]
        fname = os.path.join(logdir, basename)
        if not os.path.exists(fname):
            os.mkdir(fname)
        fname = os.path.join(logdir, basename, basename + '.ply')
        print('> Writing point cloud to {}...'.format(fname))
        data.write(fname)
    offset += step

oacc = np.sum(accu) / np.sum(freq)
accu = accu / freq
iou  = inter / union

p = np.zeros(num_classes)
r = np.zeros(num_classes)
for i in range(num_classes):
    tp = np.concatenate(tps[i], axis=0)
    fp = np.concatenate(fps[i], axis=0)
    tp = np.sum(tp)
    fp = np.sum(fp)
    p[i] = tp / (tp + fp)
    r[i] = tp / total[i]

perf = {
    'accuracy': list(accu),
    'IoU': list(iou),
    'precision': list(p),
    'recall': list(r)
}

print('> Overall accuracy: {:.3f}'.format(oacc))
print('> Mean accuracy: {:.3f}'.format(np.mean(accu)))
print('> Mean IoU: {:.3f}'.format(np.mean(iou)))
print('> Mean precision: {:.3f}'.format(np.mean(p)))
print('> Mean recall: {:.3f}'.format(np.mean(r)))

fname = os.path.join(logdir, 'eval.json')
print('> Writing report to {}...'.format(fname))
with open(fname, 'w') as fp:
    json.dump(perf, fp, indent=4)
