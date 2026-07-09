# Reproduces Figure 6 (left-hand image) of:
# "A General Approach to Visualizing Uncertainty in Statistical Graphics"
# B. Petek, D. Nabergoj, E. Strumbelj. IEEE TVCG.
#
# The figure shows 95% confidence intervals for mean bill length for each
# penguin species, generated with bootplot (n = 39 samples).
# Runs without parameters: python reproduce_figure6.py
# Output: mean_estimate_95.png (in the current working directory)

# Keep matplotlib's font cache inside this directory instead of ~/.matplotlib,
# so running this script leaves no files outside the replicability folder.
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mpl-cache'))

# Computing
import numpy as np
import pandas as pd

# Plotting
import matplotlib
matplotlib.use("cairo")  # essential: figures in the paper use the cairo backend
import matplotlib.pyplot as plt
from bootplot import bootplot

# Styling
dpi = 300
plt.rcParams['figure.dpi'] = dpi
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.size'] = 12

# Data (Palmer penguins, CC0; included in this repository)
csv_file_path = 'penguins.csv'
df = pd.read_csv(csv_file_path)
penguins = df.dropna()
penguins_bill_length = penguins[['species', 'bill_length_mm']]


# Base visualization (statistical graphic without uncertainty)
def mean_estimate(data_subset, data_full, ax):
        colors = dict(zip(['Adelie', 'Chinstrap', 'Gentoo'], ['#FF8C00', '#A020F0', '#008B8B']))
        means = data_subset.groupby('species')['bill_length_mm'].mean().reindex(colors)
        for i, (species, mean) in enumerate(means.items()):
            if pd.notna(mean):
                ax.scatter(mean, i, s=10, c=colors[species], antialiased=False,marker='s',zorder=10)
        ax.set(
            xlim=(33, 55), xticks=np.arange(33, 58, 5), xlabel='Bill Length',
            ylim=(-0.5, 2.5), yticks=[0, 1, 2], ylabel=''
        )
        ax.set_yticklabels(['Adelie', 'Chinstrap', 'Gentoo'], ha='right')

        ax.grid(axis='both', linewidth=0.5, color='#E8E8E8', zorder=-50)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.get_figure().set_constrained_layout(True)


# Fix the random seed so every run draws identical resamples (deterministic output)
np.random.seed(15)

# Apply bootplot (n = 39 samples for 95% coverage)
mat = bootplot(mean_estimate,
        penguins_bill_length,
        m=39,
        output_image_path=f'mean_estimate_95.png',
        output_size_px = (dpi*3, dpi*1.5),
        verbose=True)

print("Done. Figure written to mean_estimate_95.png")
