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
sns.set()


def main():
    # rc_folder_1 = os.path.join(
    #     "test-vis", "RC-Curve-Data", "CIFAR-Normal-2"
    # )
    # with open(os.path.join(rc_folder_1, "coverage.pkl"), 'rb') as fp:
    #     c1 = pickle.load(fp)
    # with open(os.path.join(rc_folder_1, "risk.pkl"), 'rb') as fp:
    #     r1 = pickle.load(fp)

    # rc_folder_2 = os.path.join(
    #     "test-vis", "RC-Curve-Data", "CIFAR-Margin-1"
    # )
    # with open(os.path.join(rc_folder_2, "coverage.pkl"), 'rb') as fp:
    #     c2 = pickle.load(fp)
    # with open(os.path.join(rc_folder_2, "risk.pkl"), 'rb') as fp:
    #     r2 = pickle.load(fp)
    rc_folder = os.path.join(
        "test-vis", "RC-Curve-Data"
    )
    case_names = [f for f in os.listdir(rc_folder)]
    for case in case_names:
        exp_folder = os.path.join(rc_folder, case)
        with open(os.path.join(exp_folder, "coverage.pkl"), 'rb') as fp:
            c1 = pickle.load(fp)
        with open(os.path.join(exp_folder, "risk.pkl"), 'rb') as fp:
            r1 = pickle.load(fp)

        plt.plot(c1, r1, label=case)

    plt.xlabel("coverage")
    plt.ylabel("SC risk")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    print("Completed")