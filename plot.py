import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--root', help='path to root directory')
parser.add_argument('--metric', help='metric to plot')
args = parser.parse_args()

root = args.root
metric = args.metric

fname = os.path.join(root, 'metadata', 'classes.txt')
classes = [line.strip() for line in open(fname, 'r')]
classes += ['mean']

runs = [d for d in os.listdir('logs') if
        os.path.isdir(os.path.join('logs', d))]

x = np.arange(len(classes))
markers = ['o', 'v', '^', '>', '8', 's']
plt.figure(figsize=(6, 4))

for i, run in enumerate(runs):
    logdir = os.path.join('logs', run)
    fname = os.path.join(logdir, 'eval.json')
    if not os.path.exists(fname): continue

    perf = json.load(open(fname))
    y = perf[metric]
    y += [np.mean(y)]
    plt.scatter(x, y, zorder=3, marker=markers[i], label=run)

plt.grid(True, linestyle='--')
plt.ylim([0.0, 1.0])
plt.yticks(np.linspace(0, 1, num=11))
plt.xticks(x, classes, rotation=30)
plt.tight_layout()
plt.legend()
plt.show()
