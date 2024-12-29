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
    
metrics_dict = {
    'methods': ['1D MAP', 'MART', 'U-Net'],
    'parameter': 'Intensity',
    'swept_param': 'K (Num. of Orders)',
    'metric': 'RMSE',
    'array': [
        [0.012, 0.011, 0.011, 0.010],
        [0.013, 0.012, 0.011, 0.011],
        [0.010, 0.010, 0.009, 0.009],
    ],
    'swept_arr': [2,3,4,5]
}

def comparison_sweep(*, methods=None, parameter=None, swept_param=None, metric=None,
array=None, swept_arr=None):
    markers = ['s','d','o']
    fig = plt.figure(figsize=(3.2,2.4))
    for i, method in enumerate(methods):
        plt.plot(swept_arr, array[i], label=method, marker=markers[i], linestyle='dashed')
    plt.xlabel(swept_param)
    plt.ylabel('{} {}'.format(parameter,metric))
    plt.legend()
    plt.grid(which='both', axis='both')
    plt.tight_layout()
    return fig