import os
import h5py
import numpy as np
import torch.utils.data as data


class S3DIS(data.Dataset):
    def __init__(self, root, training=True):
        self.root = root
        self.split = 'train.txt' if training else 'test.txt'
        self.flist = os.path.join(self.root, 'metadata', self.split)
        self.rooms = [line.strip() for line in open(self.flist)]
        # Load all data into memory
        self.coords = []
        self.points = []
        self.labels = []
        for fname in self.rooms:
            fin = h5py.File(os.path.join(self.root, 'h5', fname))
            self.coords.append(fin['coords'][:])
            self.points.append(fin['points'][:])
            self.labels.append(fin['labels'][:])
            fin.close()
        self.coords = np.concatenate(self.coords, axis=0)
        self.points = np.concatenate(self.points, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        # Post-processing
        self.dataset_size = self.points.shape[0]
        self.num_points = self.points.shape[1]
        for i in range(self.dataset_size):
            self.labels[i,:,1] = np.unique(self.labels[i,:,1], False, True)[1]
        self.max_instances = np.amax(self.labels[:,:,1]) + 1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        masks = np.zeros((self.num_points, self.max_instances), dtype=np.float32)
        masks[np.arange(self.num_points), self.labels[i,:,1]] = 1
        return {
            'coords': self.coords[i],
            'points': self.points[i],
            'labels': self.labels[i],
            'masks': masks,
            'size': np.unique(self.labels[i,:,1]).size
        }
