import matplotlib.pyplot as plt
import numpy as np

def barplot_group(data, labels_gr, labels_mem, ylabel, title, savedir=None):
    """
    Given a 2d 'data' where the first dimension is the different groups with the
    same dimension as `labels`, plots a grouped bar plot.
    """
    width = 1
    delta = 0.1
    x = np.arange(len(labels_gr)) * (data.shape[1]*(width+delta) + 10*delta)

    fig, ax = plt.subplots()
    rects = []
    for i in range(data.shape[1]):
        rects.append(ax.bar(x+i*(width+delta), data[:,i], width, label=labels_mem[i]))
        ax.bar_label(rects[i], padding=3, fmt='%.2f')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x+(data.shape[1]-1)*(width+delta)/2, labels_gr)
    ax.legend()

    fig.tight_layout()
    plt.grid(which='both', axis='y')

    if savedir is not None:
        plt.savefig(savedir)
    else:
        plt.show()
    