import os
import json
import h5py
import datetime
import argparse
import numpy as np
import torch
import torch.utils.data as data
from sklearn.cluster import MeanShift, estimate_bandwidth

from loaders import S3DIS
from models import MTPNet


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='path to the json config file')
parser.add_argument('--logdir', help='path to the logging directory')
args = parser.parse_args()

config = args.config
logdir = args.logdir
args = json.load(open(config))

device = 'cuda:0'
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

pred = {'semantics': [], 'instances': []}
with torch.no_grad():
    for i, batch in enumerate(loader):
        points = batch['points'].to(device)
        size = batch['size']
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

        pred['semantics'].append(semantics)
        pred['instances'].append(instances)

        now = datetime.datetime.now()
        log = '{} | Batch [{:04d}/{:04d}] |'
        log = log.format(now.strftime("%c"), i, len(loader))
        print(log)

pred['semantics'] = np.concatenate(pred['semantics'], axis=0)
pred['instances'] = np.concatenate(pred['instances'], axis=0)

fname = os.path.join(logdir, 'pred.npz')
print('> Saving predictions to {}...'.format(fname))
np.savez(fname, **pred)
