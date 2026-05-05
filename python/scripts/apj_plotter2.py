import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# DATA DEFINITION
# =============================================================================
K_vals = [2, 3, 4, 5]
gamma_vals = ['10', '20', '30', '∞']
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
            'RMSE': [[78.9, 63.4, 91.6, 67.0], [2.220, 1.841, 1.847, 1.766], [1.381, 1.300, 1.252, 1.057]],
            'Bias': [[-46.6, -39.5, -64.1, -44.7], [-0.161, 0.140, -0.074, 0.272], [0.032, -0.108, 0.210, 0.035]]
        },
        'MART': {
            'RMSE': [[379.6, 376.7, 370.8, 366.5], [3.503, 3.233, 3.199, 3.165], [4.600, 4.660, 4.763, 4.916]],
            'Bias': [[345.9, 343.2, 337.7, 334.2], [0.110, 0.088, 0.068, 0.073], [4.301, 4.390, 4.493, 4.670]]
        },
        '1D MAP': {
            'RMSE': [[378.4, 374.6, 368.4, 364.0], [3.532, 3.222, 3.690, 4.139], [3.764, 3.845, 4.289, 4.980]],
            'Bias': [[344.5, 341.0, 334.9, 331.1], [0.395, 0.751, 1.298, 1.941], [3.414, 3.505, 3.966, 4.616]]
        }
    },
    'Gamma_sweep': {
        'U-Net': {
            'RMSE': [[84.4, 110.5, 96.8, 63.4], [2.695, 2.615, 2.138, 1.841], [1.574, 1.554, 1.433, 1.300]],
            'Bias': [[-17.1, -66.7, -68.1, -39.5], [-0.071, 0.547, 0.184, 0.140], [0.100, 0.114, 0.123, -0.108]]
        },
        'MART': {
            'RMSE': [[425.9, 389.9, 382.4, 376.7], [4.920, 3.746, 3.472, 3.233], [5.124, 4.779, 4.712, 4.660]],
            'Bias': [[342.4, 343.2, 343.0, 343.2], [0.086, 0.090, 0.091, 0.088], [4.261, 4.357, 4.375, 4.390]]
        },
        '1D MAP': {
            'RMSE': [[378.8, 375.3, 375.5, 374.6], [3.753, 3.461, 3.553, 3.222], [3.529, 3.593, 3.586, 3.845]],
            'Bias': [[341.1, 341.0, 341.1, 341.0], [0.299, 0.310, 0.879, 0.751], [3.149, 3.226, 3.202, 3.505]]
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
        ax.set_xlabel(config['x_title'], labelpad=10)
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
    Patch(facecolor='gray', alpha=0.25, edgecolor='gray', linewidth=1.5, label='RMSE'),
    Patch(facecolor='gray', alpha=1.0, label='Bias')
]
leg2 = fig.legend(handles=metric_handles, loc='upper center', bbox_to_anchor=(0.7, 0.98), 
                  ncol=2, frameon=False, fontsize=13)

# Add legends to figure
fig.add_artist(leg1)

# Adjust layout so the legends don't overlap the plots
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

savepath = '/home/kamo/resources/slitless/figures/apj26_post_revision/'
fig.savefig(savepath+'bar_charts.png', transparent=True, dpi=300)