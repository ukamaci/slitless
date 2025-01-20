import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

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

def sincosgrid(grid_size=64, tx=1, ty=1):
    grid_size = 64

    # Generate the x and y values for a 2D grid
    x = np.linspace(0, 2 * np.pi * tx, grid_size)
    y = np.linspace(0, 2 * np.pi * ty, grid_size)
    X, Y = np.meshgrid(x, y)

    # Compute the function cos(x) * sin(y)
    return np.cos(X) * np.sin(Y)

def uiuc_i():
    # Create a blank image with values between 0 and 1
    width, height = 64, 64
    image_data = np.zeros((height, width), dtype=float)

    # Create a PIL image to draw the letters
    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)

    # Draw a large "I" in the center with a boundary
    large_i_width = 27
    large_i_height = 45
    bar_thickness = 9
    border_thickness = 1
    center_x = width // 2
    center_y = height // 2

    # Border color for better visibility
    border_color = 128
    fill_color = 255

    # Draw outer boundary for the top bar
    draw.rectangle([(center_x - large_i_width // 2 - border_thickness, center_y - large_i_height // 2 - border_thickness),
                    (center_x + large_i_width // 2 + border_thickness, center_y - large_i_height // 2 + bar_thickness + border_thickness)], fill=border_color)

    # Draw outer boundary for the vertical stem
    draw.rectangle([(center_x - bar_thickness // 2 - border_thickness, center_y - large_i_height // 2 + bar_thickness - border_thickness),
                    (center_x + bar_thickness // 2 + border_thickness, center_y + large_i_height // 2 - bar_thickness + border_thickness)], fill=border_color)

    # Draw outer boundary for the bottom bar
    draw.rectangle([(center_x - large_i_width // 2 - border_thickness, center_y + large_i_height // 2 - bar_thickness - border_thickness),
                    (center_x + large_i_width // 2 + border_thickness, center_y + large_i_height // 2 + border_thickness)], fill=border_color)

    # Draw top horizontal bar
    draw.rectangle([(center_x - large_i_width // 2, center_y - large_i_height // 2),
                    (center_x + large_i_width // 2, center_y - large_i_height // 2 + bar_thickness)], fill=fill_color)

    # Draw vertical stem
    draw.rectangle([(center_x - bar_thickness // 2, center_y - large_i_height // 2 + bar_thickness),
                    (center_x + bar_thickness // 2, center_y + large_i_height // 2 - bar_thickness)], fill=fill_color)

    # Draw bottom horizontal bar
    draw.rectangle([(center_x - large_i_width // 2, center_y + large_i_height // 2 - bar_thickness),
                    (center_x + large_i_width // 2, center_y + large_i_height // 2)], fill=fill_color)

    # Normalize the image to be between 0 and 1
    image_data = np.array(image) / 255.0

    return image_data

def uiuc_im():
    im = sincosgrid(64,9,9)+3*uiuc_i()
    im = np.repeat(im[None], 3, axis=0)
    imax= im.max()
    imin = im.min()
    im[0] = (im[0]-imin)/(imax-imin)
    im[1] = (im[1]-imin)/(imax-imin)*2 - 1
    im[2] = (im[2]-imin)/(imax-imin) + 1
    return im
    
# metrics_dict = {
#     'methods': ['1D MAP', 'MART', 'U-Net'],
#     'parameter': 'Intensity',
#     'swept_param': 'K (Num. of Orders)',
#     'metric': 'RMSE',
#     'array': [
#         [0.012, 0.011, 0.011, 0.010],
#         [0.013, 0.012, 0.011, 0.011],
#         [0.010, 0.010, 0.009, 0.009],
#     ],
#     'swept_arr': [2,3,4,5]
# }

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