import os
import math
import matplotlib.pyplot as plt

from plot import (
    main,
    gammas,
    calibrate,
    invert_kl,
    ln,
)


def build_q_variants(betas, av_bcetrain, samplesize, calibration_factor):
    """
    Build three Q/n variants:
    1) Harel Bound: Q = beta * av_bcetrain[0]
    2) Gamma Bound: Q = gamma_i where gamma = gammas(..., factor=1)
    3) Calibrated Gamma Bound: Q = calibration_factor * gamma_i
    """
    if not betas or not av_bcetrain:
        raise ValueError("betas and av_bcetrain must be non-empty")

    gamma_base = gammas(betas, av_bcetrain, factor=1.0)

    q_over_n_harel = [(beta * av_bcetrain[0]) / samplesize for beta in betas]
    q_over_n_gamma = [g / samplesize for g in gamma_base]
    q_over_n_gamma_cal = [(calibration_factor * g) / samplesize for g in gamma_base]

    return q_over_n_harel, q_over_n_gamma, q_over_n_gamma_cal, gamma_base


def bounds_from_q_over_n(q_over_n_values, samplesize, delta=0.01):
    """
    Convert Q/n values to KL-budget values used by invert_kl.
    """
    log_term = ln(2 * math.sqrt(samplesize) / delta) / samplesize
    return [q_over_n + log_term for q_over_n in q_over_n_values]


def predict01_from_bounds(av_train01, bounds):
    """
    Predict 0-1 test error bound via inverse KL, like plot.py predict01.
    """
    if len(av_train01) != len(bounds):
        raise ValueError("av_train01 and bounds must have the same length")

    preds = []
    for i in range(len(bounds)):
        preds.append(invert_kl(av_train01[i], bounds[i]))

    if preds:
        preds[0] = 0.5

    return preds


def show_q_over_n_variants(betas, q_over_n_harel, q_over_n_gamma, q_over_n_gamma_cal, samplesize, randomfilename):
    """
    Plot the three Q/n variants.
    """
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "lines.linewidth": 2.5,
        "figure.figsize": (10, 7),
        "axes.grid": False,
    })

    fig, ax = plt.subplots()
    ax.semilogx()

    ax.plot(betas[1:], q_over_n_harel[1:], "o-k", markersize=7, label="Harel et al. Bound")
    ax.plot(betas[1:], q_over_n_gamma[1:], "s-b", markersize=7, label="Our Bound")
    ax.plot(betas[1:], q_over_n_gamma_cal[1:], "^-r", markersize=7, label="Our Calibrated Bound")

    # Add vertical dashed line where beta = n (samplesize)
    ax.axvline(x=samplesize, color="gray", linestyle="--", linewidth=1.5, label=f"beta=n ({samplesize})")

    ax.set_xlabel("Beta")
    ax.set_ylabel(r"KL($\nu_{\beta}$, $\pi$) / n")
    ax.legend(frameon=True, loc="best", framealpha=0.9, edgecolor="black")

    ax.minorticks_on()
    ax.tick_params(which="minor", length=3, color="gray")
    ax.tick_params(which="major", length=6, width=1.2)

    os.makedirs("newplots", exist_ok=True)
    out_name = f"newplots/{randomfilename[:-4]}_q_over_n_variants.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.show()


def show01_three_bounds(
    betas,
    av_train01,
    av_test01,
    pred_harel,
    pred_gamma,
    pred_gamma_cal,
    plot_filename,
    y_limits=None,
    y_scale="linear",
):
    """
    Plot train/test 0-1 errors and three different predicted bounds.
    """
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "lines.linewidth": 2.5,
        "figure.figsize": (10, 7),
        "axes.grid": False,
    })

    fig, ax = plt.subplots()
    ax.semilogx()

    ax.plot(betas[1:], av_train01[1:], "o-k", markersize=7, label="Train 0-1 Error")
    ax.plot(betas[1:], av_test01[1:], "^-g", markersize=7, label="Test 0-1 Error")

    ax.plot(betas[1:], pred_harel[1:], "s-r", markersize=6, label="Bound (Harel Q)")
    ax.plot(betas[1:], pred_gamma[1:], "D-b", markersize=6, label="Bound (Gamma Q)")
    ax.plot(betas[1:], pred_gamma_cal[1:], "v-m", markersize=6, label="Bound (Calibrated Gamma Q)")

    ax.set_xlabel("Beta")
    ax.set_ylabel("0-1 Error")
    ax.set_yscale(y_scale)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.legend(frameon=True, loc="best", framealpha=0.9, edgecolor="black")

    ax.minorticks_on()
    ax.tick_params(which="minor", length=3, color="gray")
    ax.tick_params(which="major", length=6, width=1.2)

    os.makedirs("newplots", exist_ok=True)
    out_name = f"newplots/{plot_filename[:-4]}_01_three_bounds.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.show()


# CONTROL ------------------------------------------------
# display = 3 -> show Q/n variants
# display = 4 -> show 0-1 plot with three bounds

display = 3
trueLabels = 0  # 0 = random labels, 1 = true labels

# Controls for y-axis in show01_three_bounds
y_scale_three_bounds = "linear"  # "linear" or "log"
y_limits_three_bounds = (0, 1)  # Set to None to use matplotlib default

# Keep naming flow similar to plot.py
truefilename, randomfilename = (
    # "MCL2W1000SGLD8kLR001BBCE.csv",
    # "MRL2W1000SGLD8kLR001BBCE.csv",
    "CCL2W1500SGLD8kLR0005BBCE.csv",
    "CRL2W1500SGLD8kLR0005BBCE.csv",
)

# Load random-label CSV first for calibration (happens once).
betas_random, bcetrain_random, bcetest_random, train01_random, test01_random, av_bcetrain_random, av_bcetest_random, av_train01_random, av_test01_random, n_samples_random = main(randomfilename)
samplesize_random = n_samples_random[0]

# Calibrate ONCE on random labels.
factor = calibrate(betas_random, av_bcetrain_random, av_train01_random, samplesize_random, thresh=0.5)
print("calibration factor =", factor)

# Select which dataset to plot (random or true labels).
if trueLabels == 0:
    betas = betas_random
    av_bcetrain = av_bcetrain_random
    av_train01 = av_train01_random
    av_test01 = test01_random
    samplesize = samplesize_random
    plot_filename = randomfilename
else:  # trueLabels == 1
    betas, bcetrain, bcetest, train01, test01, av_bcetrain, av_bcetest, av_train01, av_test01, n_samples = main(truefilename)
    samplesize = n_samples[0]
    plot_filename = truefilename

# Compute Q variants and bounds using the selected dataset, but with calibration factor from random labels.
q_over_n_harel, q_over_n_gamma, q_over_n_gamma_cal, gamma_base = build_q_variants(
    betas, av_bcetrain, samplesize, factor
)

bounds_harel = bounds_from_q_over_n(q_over_n_harel, samplesize, delta=0.01)
bounds_gamma = bounds_from_q_over_n(q_over_n_gamma, samplesize, delta=0.01)
bounds_gamma_cal = bounds_from_q_over_n(q_over_n_gamma_cal, samplesize, delta=0.01)

pred_harel = predict01_from_bounds(av_train01, bounds_harel)
pred_gamma = predict01_from_bounds(av_train01, bounds_gamma)
pred_gamma_cal = predict01_from_bounds(av_train01, bounds_gamma_cal)

if display == 3:
    show_q_over_n_variants(betas, q_over_n_harel, q_over_n_gamma, q_over_n_gamma_cal, samplesize, plot_filename)
elif display == 4:
    show01_three_bounds(
        betas,
        av_train01,
        av_test01,
        pred_harel,
        pred_gamma,
        pred_gamma_cal,
        plot_filename,
        y_limits=y_limits_three_bounds,
        y_scale=y_scale_three_bounds,
    )
