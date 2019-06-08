import os
import json
import h5py
import datetime
import argparse
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
from sklearn.cluster import MeanShift

from loaders import *
from models import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='path to the logging directory')
parser.add_argument('--mvcrf', action='store_true', help='use MV-CRF for post-processing')
args = parser.parse_args()

logdir = args.logdir
config = os.path.join(logdir, 'config.json')
args = json.load(open(config))

device = args['device']
dataset = S3DIS(args['root'], training=False)
loader = data.DataLoader(
    dataset,
    batch_size=args['batch_size'],
    num_workers=args['num_workers'],
    pin_memory=True,
    shuffle=False
)

fname = os.path.join(logdir, 'model.pth')
print('> Loading model from {}....'.format(fname))
model = MTPNet(
    args['input_channels'],
    args['num_classes'],
    args['embedding_size']
)
model.load_state_dict(torch.load(fname))
model.to(device)
model.eval()

pdict = {'semantics': [], 'instances': []}
with torch.no_grad():
    for i, batch in enumerate(tqdm(loader, ascii=True)):
        points = batch['points'].to(device)
        labels = batch['labels']
        size   = batch['size']

        logits, embedded = model(points)
        logits = logits.cpu().numpy()
        semantics = np.argmax(logits, axis=-1)

        instances = []
        embedded = embedded.cpu().numpy()
        batch_size = embedded.shape[0]
        for b in range(batch_size):
            k = size[b].item()
            y = MeanShift(args['bandwidth'], n_jobs=8).fit_predict(embedded[b])
            instances.append(y)
        instances = np.stack(instances)

        pdict['semantics'].append(semantics)
        pdict['instances'].append(instances)

pdict['semantics'] = np.concatenate(pdict['semantics'], axis=0)
pdict['instances'] = np.concatenate(pdict['instances'], axis=0)
pdict = np.stack([pdict['semantics'], pdict['instances']], axis=-1)

fname = os.path.join(args['root'], 'metadata', 'test.txt')
flist = [line.strip() for line in open(fname)]

offset = 0
for fname in tqdm(flist, ascii=True):
    fname = os.path.join(args['root'], 'h5', fname)
    fin = h5py.File(fname)
    coords = fin['coords'][:]
    points = fin['points'][:]
    batch_size = coords.shape[0]
    num_points = coords.shape[1]

    pred = pdict[offset:offset + batch_size]
    pred = block_merge(points[:, :, 6:9], pred)
    pred = pred.reshape(-1, 2)

    if args.mvcrf:
        coords = coords.reshape(-1, 3)
        points = points.reshape(-1, 9)

        fname = os.path.join(logdir, 'pred.npz')
        data = {'coords': coords, 'points': points, 'pred': pred}
        np.savez(fname, **data)
        prog = './mvcrf {}'.format(fname)
        os.system(prog)

        data = np.load(fname)
        pred = data['pred']

    pred = pred.reshape(batch_size, num_points, 2)
    pdict[offset:offset + batch_size] = pred
    offset += batch_size

pdict = {'semantics': pdict[:, :, 0], 'instances': pdict[:, :, 1]}
fname = os.path.join(logdir, 'pred.npz')
print('> Saving predictions to {}...'.format(fname))
np.savez(fname, **pdict)
