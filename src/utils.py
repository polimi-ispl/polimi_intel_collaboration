import os
import random
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, normalize=False, cmap=plt.cm.Blues):
    """
    Plot confusion matrix.

    :param y_true: ground-truth labels
    :type y_true: list
    :param y_pred: predicted labels
    :type y_pred: list
    :param normalize: if set to True, normalise the confusion matrix.
    :type normalize: bool
    :param cmap: matplotlib cmap to be used for plot
    :type cmap:
    :return:
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = ['$\it{Real}$','$\it{Fake}$']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    fsize = 25 
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, clim=(0,1))
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=fsize)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           )
    ax.set_xlabel('$\mathrm{Predicted\;label}$', fontsize=fsize)
    ax.set_ylabel('$\mathrm{True\;label}$', fontsize=fsize)
    ax.set_xticklabels(classes, fontsize=fsize)
    ax.set_yticklabels(classes, fontsize=fsize)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format('$\mathrm{' + str(format(cm[i, j], fmt)) + '}$'),
                    ha="center", va="center",
                    fontsize=fsize,
                    color="white" if np.array(cm[i, j]) > thresh else "black")
    fig.tight_layout()

    return ax


def seed_everything(seed: int):
    """
    Set seed for everything.
    :param seed: seed value
    :type seed: int
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
