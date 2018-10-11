import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def import_phonemes():
    return sorted([x.strip() for x in open('phonemeList.txt').readlines()])


def plot_features():
    phones = import_phonemes()

    data = np.load('dataset/valdata_.npz')['data']
    x = np.concatenate([x['lmfcc'] for x in data[:200]])
    y = np.concatenate([d['target'] for d in data[:200]])
    y = [phones.index(i) for i in y]

    e = TSNE(n_components=3).fit_transform(x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(e[:, 0], e[:, 1], e[:, 2], c=y)
    plt.savefig('tsne.png')
