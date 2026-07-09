Title: A General Approach to Visualizing Uncertainty in Statistical Graphics
Authors: Bernarda Petek, David Nabergoj, Erik Strumbelj
Venue: IEEE Transactions on Visualization and Computer Graphics
Operating system: macOS 11 or newer
Code repository: https://github.com/davidnabergoj/bootplot (this submission is the reproducibility_challenge directory inside it)
Reproduced result: Figure 6, left-hand image, produced by our approach that shows 95% confidence intervals for mean bill length per penguin species, created with bootplot, n = 39 samples.

========================================================================

WHAT THIS DIRECTORY CONTAINS
------------------------------------------------------------------------
1. install_dependencies.sh -- installs all dependencies (including bootplot) and runs the Python script below.
2. reproduce_figure6.py -- Python script that runs bootplot to produce the figure.
3. penguins.csv -- the dataset used by our method to produce the figure (Palmer penguins, CC0 license).
4. README.txt -- this file: description and instructions.
5. LIABILITY_FORM.txt -- the liability form.

HOW TO RUN THE SCRIPT
------------------------------------------------------------------------
On macOS 11 or newer, open a terminal (for example, the built-in Terminal.app) and run the script by giving bash its path. For example, if you cloned the repository into your Downloads folder:

    bash ~/Downloads/bootplot/reproducibility_challenge/install_dependencies.sh

(~ is your home folder: /Users/<your-username>; adjust the path to wherever you cloned the repository). That is the only command needed. After a few minutes, the reproduced figure opens in Preview and is saved next to the script as mean_estimate_95.png. No administrator password is needed, and nothing is asked during the run, but macOS may ask for permission to access the folder (depending on the folder). For example, if the repository is in Downloads, click OK.

WHAT THE SCRIPT DOES, STEP BY STEP
------------------------------------------------------------------------
1. Downloads Miniconda from Anaconda's official server (repo.anaconda.com), automatically choosing the Apple Silicon or Intel build.
2. Installs Miniconda into the subfolder named miniconda3, inside this directory. Hence, nothing is written into system folders or the home directory, and no administrator password is ever needed.
3. Creates a Python environment in the subfolder named env containing Python 3.12, pip, and pycairo from the conda-forge channel. Figures are rendered with matplotlib's cairo backend, the backend used for the figures in the paper (it allows disabling anti-aliasing, as the paper recommends). conda-forge provides pycairo prebuilt with the cairo graphics library bundled in, so nothing is compiled on your machine.
4. Installs the bootplot library (version 0.0.18) from PyPI into that environment. Installing bootplot automatically brings its own dependencies (numpy, pandas, matplotlib, and others).
5. Runs reproduce_figure6.py with no parameters. The script reads the bundled penguins.csv, applies bootplot with n = 39 resamples, which yields 95% coverage (Table I in the paper), saves the aggregate image to mean_estimate_95.png, and opens it in Preview. On success, the final output message is: "Done. The figure was saved as mean_estimate_95.png in this directory."
