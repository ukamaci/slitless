import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# =============================================================================
# DATA DEFINITION
# =============================================================================
K_vals = [2, 3, 4, 5]
gamma_vals = ['10', '20', '30', r'$\infty$']
params = ['Intensity', 'Velocity', 'Line Width']
methods = ['U-Net', 'MART', '1D MAP']

# APJ-friendly, high-contrast, colorblind-safe palette (Okabe-Ito)
colors = {
    'U-Net': '#0072B2',  # Strong Blue
    'MART': '#D55E00',   # Vermillion / Deep Orange
    '1D MAP': '#009E73'  # Teal / Strong Green
}

data = {
    'K_sweep': {
        'U-Net': {
            'RMSE': [[64.5, 90.6, 92.9, 99.3], [2.376, 1.920, 1.690, 1.674], [1.569, 1.445, 1.330, 1.314]],
            'Bias': [[-33.9, -60.0, -70.4, -68.6], [0.652, 0.237, 0.107, 0.024], [0.019, -0.015, 0.032, -0.079]]
        },
        'MART': {
            'RMSE': [[59.2, 56.2, 56.0, 54.8], [5.962, 4.126, 3.590, 3.425], [2.030, 1.984, 2.014, 2.034]],
            'Bias': [[-5.3, -5.6, -5.8, -5.0], [4.321, 3.177, 2.297, 2.059], [0.322, 0.237, 0.360, 0.445]]
        },
        '1D MAP': {
            'RMSE': [[60.6, 60.6, 60.6, 60.6], [5.760, 5.071, 4.620, 4.204], [4.104, 3.141, 3.593, 3.758]],
            'Bias': [[18.9, 18.9, 18.9, 18.9], [1.403, 1.894, 1.537, 1.224], [0.374, 0.395, 1.394, 1.795]]
        }
    },
    'Gamma_sweep': {
        'U-Net': {
            'RMSE': [[116.5, 83.7, 108.8, 90.6], [3.289, 2.883, 2.446, 1.920], [1.799, 1.661, 1.642, 1.445]],
            'Bias': [[-69.6, -44.8, -47.5, -60.0], [0.728, 0.688, 0.525, 0.237], [0.077, -0.040, -0.006, -0.015]]
        },
        'MART': {
            'RMSE': [[159.6, 92.9, 75.1, 56.2], [10.459, 5.343, 4.559, 4.126], [6.202, 3.429, 2.701, 1.984]],
            'Bias': [[-20.6, -8.6, -7.1, -5.6], [-4.654, 1.049, 2.173, 3.177], [1.611, 0.686, 0.378, 0.237]]
        },
        '1D MAP': {
            'RMSE': [[148.6, 92.5, 76.9, 60.6], [9.241, 8.587, 7.329, 5.071], [3.028, 3.084, 3.274, 3.141]],
            'Bias': [[18.9, 18.9, 18.8, 18.9], [2.874, 2.913, 2.893, 1.894], [2.044, 1.870, 1.328, 0.395]]
        }
    }
}

# =============================================================================
# PLOTTING SETUP
# =============================================================================
# Set larger font sizes for publication readability
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11})

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

x_k = np.arange(len(K_vals))
x_g = np.arange(len(gamma_vals))

w_rmse = 0.25  # Wide outer bar
w_bias = 0.12  # Narrow inner bar

# Subplot Configuration
configs = [
    {'row': 0, 'sweep': 'K_sweep', 'x_arr': x_k, 'x_labels': K_vals, 'x_title': 'Number of Projections (K)'},
    {'row': 1, 'sweep': 'Gamma_sweep', 'x_arr': x_g, 'x_labels': gamma_vals, 'x_title': 'SNR ($\gamma$)'}
]

# Physical Units for Y-axes
y_labels = [
    r'Intensity Error ($\mathrm{erg \ cm^{-2} \ s^{-1} \ sr^{-1}}$)', 
    r'Velocity Error ($\mathrm{km \ s^{-1}}$)', 
    r'Line Width Error ($\mathrm{km \ s^{-1}}$)'
]

# =============================================================================
# DRAW PANELS
# =============================================================================
for config in configs:
    row = config['row']
    sweep = config['sweep']
    
    for col, param in enumerate(params):
        ax = axes[row, col]
        
        # Add titles and labels
        if row == 0: 
            ax.set_title(param, fontsize=15, pad=15, fontweight='bold')
        ax.set_xlabel(config['x_title'], labelpad=2)
        if col == 0: 
            ax.set_ylabel(y_labels[col], labelpad=10)
        else:
            ax.set_ylabel(y_labels[col], labelpad=10)

        # Plot Data
        for i, method in enumerate(methods):
            offset = (i - 1) * w_rmse
            color = colors[method]
            
            # RMSE Bar (Wide, pale background, solid edge)
            ax.bar(config['x_arr'] + offset, data[sweep][method]['RMSE'][col], w_rmse, 
                   color=color, alpha=0.25, edgecolor=color, linewidth=1.5, zorder=2)
            
            # Bias Bar (Narrow, fully solid inside)
            ax.bar(config['x_arr'] + offset, data[sweep][method]['Bias'][col], w_bias, 
                   color=color, alpha=1.0, zorder=3)
            
        # Formatting
        ax.set_xticks(config['x_arr'])
        ax.set_xticklabels(config['x_labels'])
        ax.axhline(0, color='black', linewidth=1.2, zorder=1) # Strong Zero Line
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# =============================================================================
# DUAL CUSTOM LEGENDS (Methods + Metrics)
# =============================================================================
# 1. Legend for the Methods (Colors)
method_handles = [Patch(facecolor=colors[m], edgecolor=colors[m], label=m) for m in methods]
leg1 = fig.legend(handles=method_handles, loc='upper center', bbox_to_anchor=(0.35, 0.98), 
                  ncol=3, frameon=False, fontsize=13)

# 2. Legend for the Metrics (Shapes)
metric_handles = [
    Line2D([0], [0], color='gray', alpha=0.3, linewidth=14, solid_capstyle='butt', label='RMSE (Wide Bar)'),
    Line2D([0], [0], color='gray', alpha=1.0, linewidth=5, solid_capstyle='butt', label='Bias (Inner Bar)')
]
leg2 = fig.legend(handles=metric_handles, loc='upper center', bbox_to_anchor=(0.75, 0.98), 
                  ncol=2, frameon=False, fontsize=13)

# Add legends to figure
fig.add_artist(leg1)

# Adjust layout so the legends don't overlap the plots and rows are clearly separated
plt.tight_layout(rect=[0, 0, 1, 0.92], h_pad=3.0)
plt.show()

savepath = '/home/kamo/resources/slitless/figures/apj26_post_revision/'
fig.savefig(savepath+'bar_charts.png', transparent=True, dpi=300)