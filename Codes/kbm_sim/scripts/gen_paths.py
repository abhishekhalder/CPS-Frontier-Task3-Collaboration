#
# Generate paths to be used by the KBM simulator. These paths are to all start
# on the line x=0 and follow a Gaussian process out to x=X
#
# Original code from: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py
#
# Author: Georgiy Antonovich Bondar
# Date  : 05-17-2023
#
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import numpy as np
import pdb

# ----- Parameters ----- #
n_paths  = 12       # Number of paths to generate
path_len = 10.0     # Length of each path
variance = 10.0     # A proxy for variance of path generation
x_step   = 0.05     # Interval between path x coordinates
v_des    = 12.0     # Desired velocity at each timestep (used by PID)
seed     = None     # Change to an int for repeatable results

outfile_prefix = "../sim_paths/path_wypts_"
outfile_suffix = ".txt"
# ---------------------- #

def gen_plot_gpr_samples(gpr_model, n_samples, ax):
    x = np.linspace(0, path_len, int(path_len/x_step))
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples, random_state=seed)

    # Plot and write paths
    for idx, y in enumerate(y_samples.T):
        ax.plot(
            x,
            y,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
        with open(outfile_prefix + str(idx) + outfile_suffix, "w") as f:
            for i in range(len(x)):
                f.write(str(x[i]) + ", " + str(y[i]) + ", " + str(v_des) + "\n")

    # Plot mean + stdev
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-10, 10])


kernel = variance * RBF(length_scale=3.0, length_scale_bounds=(1e-1, path_len))
gpr = GaussianProcessRegressor(kernel=kernel)

# Plot paths
fig, axs = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(10, 8))
gen_plot_gpr_samples(gpr, n_samples=n_paths, ax=axs)
axs.set_title("KBM Paths")
fig.savefig("../plots/generated_paths.png")
# plt.show()


