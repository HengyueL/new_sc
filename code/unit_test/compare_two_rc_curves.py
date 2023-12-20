import pickle
import os
import sys
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
path2 = os.path.join(dir_path, "..")
sys.path.append(path2)
path3 = os.path.join(dir_path, "code")
sys.path.append(path3)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
sns.set()
COLORS = list(mcolors.TABLEAU_COLORS)

def main():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))  # ax[0] --- RC curves; ax[1] --- full coverage acc
    rc_folder = os.path.join(
        "test-vis", "RC-Curve-Data"
    )

    acc_plot_x, acc_plot_x_names = [], []

    case_names = [f for f in os.listdir(rc_folder)]
    for color_idx, case in enumerate(case_names):
        exp_folder = os.path.join(rc_folder, case)
        with open(os.path.join(exp_folder, "coverage.pkl"), 'rb') as fp:
            c1 = pickle.load(fp)
        with open(os.path.join(exp_folder, "risk.pkl"), 'rb') as fp:
            r1 = pickle.load(fp)
        with open(os.path.join(exp_folder, "full_cvg_acc.pkl"), 'rb') as fp:
            acc_dict = pickle.load(fp)

            acc_values = []
            if len(acc_plot_x) < 1:
                for idx, name in enumerate(acc_dict.keys()):
                    acc_plot_x.append(idx+1)
                    acc_plot_x_names.append(name)
                    acc_values.append(acc_dict[name])
            else:
                for name in acc_plot_x_names:
                    acc_values.append(acc_dict[name])
        if "stopgrad" in case.lower():
            ls, marker = "dotted", None
        elif "ce" in case.lower():
            ls, marker = "solid", None
        else:
            ls, marker = "dashed", None
        ax[0].plot(c1[0:-1:40], r1[0:-1:40], label=case, lw=3, marker=marker, ls=ls, alpha=0.7, color=COLORS[color_idx%10])
        ax[1].scatter(acc_plot_x, acc_values, label=case, lw=4, alpha=0.8, color=COLORS[color_idx%10])

    ax[0].set_xlabel("coverage")
    ax[0].set_ylabel("SC risk")
    ax[0].set_title("RC curves")
    ax[0].legend()

    ax[1].set_title("Full coverage acc. (Per Corruption Type)")
    ax[1].legend()
    ax[1].set_ylabel("Prediction accuracy")
    # Set number of ticks for x-axis
    ax[1].set_xticks(acc_plot_x)
    # Set ticks labels for x-axis
    ax[1].set_xticklabels(acc_plot_x_names, rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
    print("Completed")