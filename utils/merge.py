import numpy as np
import scipy.stats as stats


def block_merge(coords, pred):
    stride = 0.005
    shape  = np.array([200, 200, 200])
    semantic = np.ones(shape + 1) * -1
    instance = np.ones(shape + 1) * -1
    semantic = semantic.astype(np.int32)
    instance = instance.astype(np.int32)

    batch_size = coords.shape[0]
    num_points = coords.shape[1]
    coords = coords / stride
    coords = coords.astype(np.int32)

    for b in range(batch_size):
        k = np.amax(pred[b, :, 1]) + 1
        overlap = np.zeros([k, 300])
        # find label for each group
        modes = {}
        sizes = {}
        for gid in range(k):
            indices = (pred[b, : , 1] == gid)
            mode = stats.mode(pred[b, indices, 0])[0]
            modes[gid] = int(mode)
            sizes[gid] = np.sum(indices)
        # calculate overlap between blocks
        for i in range(num_points):
            x, y, z = coords[b, i]
            gid = pred[b, i, 1]
            if instance[x, y, z] >= 0 and semantic[x, y, z] == modes[gid]:
                overlap[gid, instance[x, y, z]] += 1
        label = np.argmax(overlap, axis=1)
        n = np.amax(instance)
        for gid in range(k):
            count = np.amax(overlap[gid])
            if count < 7 and sizes[gid] > 30:
                # create new instance
                n += 1
                label[gid] = n
        for i in range(num_points):
            x, y, z = coords[b, i]
            gid = pred[b, i, 1]
            if gid >= 0 and instance[x, y, z] < 0:
                instance[x, y, z] = label[gid]
                semantic[x, y, z] = modes[gid]

    for b in range(batch_size):
        for i in range(num_points):
            x, y, z = coords[b, i]
            pred[b, i, 0] = semantic[x, y, z]
            pred[b, i, 1] = instance[x, y, z]
    return pred
