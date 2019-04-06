import os
import json
import datetime
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from collections import defaultdict

from loaders import S3DIS
from models import MTPNet
from losses import NLLLoss, DiscriminativeLoss


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='path to the json config file')
parser.add_argument('--logdir', help='path to the logging directory')
args = parser.parse_args()

config = args.config
logdir = args.logdir
args = json.load(open(config))
if not os.path.exists(logdir):
    os.mkdir(logdir)

device = args['device']
dataset = S3DIS(args['root'], training=True)
loader = data.DataLoader(
    dataset,
    batch_size=args['batch_size'],
    num_workers=args['num_workers'],
    pin_memory=True,
    shuffle=True
)

model = MTPNet(
    args['input_channels'],
    args['num_classes'],
    args['embedding_size']
)
model.to(device)

parameters = model.parameters()
optimizer = optim.SGD(
    parameters,
    lr=args['learning_rate'],
    momentum=args['momentum'],
    weight_decay=args['weight_decay']
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    args['step_size'],
    gamma=args['decay_rate']
)

criterion = {}
criterion['discriminative'] = DiscriminativeLoss(
    args['delta_d'],
    args['delta_v']
)
criterion['nll'] = NLLLoss()
criterion['discriminative'].to(device)
criterion['nll'].to(device)

best_loss = np.Inf
for epoch in range(args['epochs']):
    start = datetime.datetime.now()
    scheduler.step()
    scalars = defaultdict(list)

    model.train()
    for i, batch in enumerate(loader):
        points = batch['points'].to(device)
        labels = batch['labels'].to(device)
        masks  = batch['masks'].to(device)
        size   = batch['size']

        loss = 0
        logits, embedded = model(points)
        loss += criterion['nll'](logits, labels[:,:,0])
        loss += criterion['discriminative'](embedded, masks, size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scalars['loss'].append(loss)
        now = datetime.datetime.now()
        log = '{} | Batch [{:04d}/{:04d}] | loss: {:.4f} |'
        log = log.format(now.strftime("%c"), i, len(loader), loss.item())
        print(log)

    summary = {}
    now = datetime.datetime.now()
    duration = now - start
    log = '> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |'
    log = log.format(now.strftime("%c"), epoch, args['epochs'], duration.total_seconds())
    for m, v in scalars.items():
        summary[m] = torch.stack(v).mean()
        log += ' {}: {:.4f} |'.format(m, summary[m].item())

    if summary['loss'] < best_loss:
        best_loss = summary['loss']
        fname = os.path.join(logdir, 'model.pth')
        print('> Saving model to {}...'.format(fname))
        torch.save(model.state_dict(), fname)

    log += ' best: {:.4f} |'.format(best_loss)

    fname = os.path.join(logdir, 'train.log')
    with open(fname, 'a') as fp:
        fp.write(log + '\n')

    print(log)
    print('--------------------------------------------------------------------------------')
