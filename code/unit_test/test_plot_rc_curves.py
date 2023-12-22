import os
import sys
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
path2 = os.path.join(dir_path, "..")
sys.path.append(path2)
path3 = os.path.join(dir_path, "code")
sys.path.append(path3)

import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
import torch.nn.functional as F
import random
from numpy.linalg import pinv
from sklearn.covariance import EmpiricalCovariance
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from utils.general import clear_terminal_output
from copy import deepcopy
sns.set()
COLORS = list(mcolors.TABLEAU_COLORS)
CF_METHOD_STR_LIST = []
DATASET_NAME_LIST = []
PLOT_SYMBOL_DICT = {}


def RC_curve(residuals, confidence):

    curve = []
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    cov = len(temp1)
    acc = sum(temp1)
    curve.append((cov/ m, acc / len(temp1)))
    for i in range(0, len(idx_sorted)-1):
        cov = cov-1
        acc = acc-residuals[idx_sorted[i]]
        curve.append((cov / m, acc /(m-i)))
    
    # AUC = sum([a[1] for a in curve])/len(curve)
    # err = np.mean(residuals)
    # kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    # EAURC = AUC-kappa_star_aurc
    # return curve, AUC, EAURC

    curve = np.asarray(curve)
    coverage, risk = curve[:, 0], curve[:, 1]
    return coverage, risk


def select_RC_curve_points(coverage_array, risk_array, n_plot_points=40,  min_n_samples=-10):

    plot_interval = len(coverage_array) // n_plot_points
    coverage_plot, risk_plot = coverage_array[0::plot_interval].tolist(), risk_array[0::plot_interval].tolist()
    coverage_plot.append(coverage_array[min_n_samples])
    risk_plot.append(risk_array[min_n_samples])
    return coverage_plot, risk_plot


def plot_rc_curve_demo(total_scores_dict, total_residuals_dict, fig_name, case_name="in-d", method_name_list=None):
    coverage_dict, risk_dict = {}, {}

    if method_name_list is None:
        method_name_list = CF_METHOD_STR_LIST

    if case_name in ["in-d"]:
        for method_name in method_name_list:
            coverage, sc_risk = RC_curve(
                total_residuals_dict[case_name][method_name], total_scores_dict[case_name][method_name]
            )
            coverage_dict[method_name] = coverage
            risk_dict[method_name] = sc_risk
    elif case_name == "all":
        residuals_dict, scores_dict = {}, {}
        for method_name in method_name_list:
            residuals_dict[method_name] = []
            scores_dict[method_name] = []
            for dataset_name in DATASET_NAME_LIST:
                residuals = total_residuals_dict[dataset_name][method_name]
                scores = total_scores_dict[dataset_name][method_name]
                residuals_dict[method_name].append(residuals)
                scores_dict[method_name].append(scores)
            residuals_dict[method_name] = np.concatenate(residuals_dict[method_name], axis=0)
            scores_dict[method_name] = np.concatenate(scores_dict[method_name], axis=0)
            coverage, sc_risk = RC_curve(
                residuals_dict[method_name], scores_dict[method_name]
            )
            coverage_dict[method_name] = coverage
            risk_dict[method_name] = sc_risk
    else:
        raise RuntimeError("Somthing Happened")

    # === Plot RC Curve ===
    plot_n_points = 30
    min_num_samples = -100
    save_path = fig_name
    line_width = 4
    markersize = 8
    alpha = 0.5

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    font_size = 19
    tick_size = 20

    y_min = 0
    y_max = 0
    for method_name in method_name_list:
        coverage_plot, sc_risk_plot = coverage_dict[method_name], risk_dict[method_name]
        x_plot, y_plot = select_RC_curve_points(coverage_plot, sc_risk_plot, plot_n_points, min_num_samples)
        y_max, y_min = max(y_plot[0], y_max), min(np.amin(y_plot), y_min)
        # y_max, y_min = max(np.amax(y_plot), y_max), min(np.amin(y_plot), y_min)
        plot_settings = PLOT_SYMBOL_DICT[method_name]
        ax.plot(
            x_plot, y_plot,
            label=plot_settings[2], lw=line_width, alpha=alpha,
            color=COLORS[plot_settings[0]], marker=plot_settings[1], ls=plot_settings[3], markersize=markersize
        )
    # ax.legend(
    #     loc='upper center', bbox_to_anchor=(0.35, 1.25),
    #     ncol=2, fancybox=True, shadow=False, fontsize=font_size, framealpha=0.3
    # )
    # COLOR_IDX = 3
    # if coverage_list is not None:
    #     for coverage in coverage_list:
    #         ax.vlines(x=coverage, ymin=y_min-1, ymax=1.15*y_max, color=COLORS[COLOR_IDX], ls="dashed", lw=2, alpha=0.65)
    #         COLOR_IDX += 1

    ax.legend(
        loc='lower left', bbox_to_anchor=(-0.25, 1, 1.25, 0.2), mode="expand", 
        borderaxespad=0,
        ncol=3, fancybox=True, shadow=False, fontsize=font_size, framealpha=0.3
    )
    ax.tick_params(axis='x', which='major', colors='black', labelsize=tick_size)
    ax.tick_params(axis='y', which='major', colors='black', labelsize=tick_size)
    ax.set_ylabel(r"Selection Risk", fontsize=font_size)
    ax.set_xlabel(r"Coverage", fontsize=font_size)
    ax.set_ylim([y_min-0.05*y_max, 1.10*y_max])
    # ax.set_xticks([0, 0.5, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim([-0.02, 1.05])
    # ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_yticks([y_max/2, y_max])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return coverage_dict, risk_dict


def calculate_residual(pred, label):
    pred_tensor = torch.from_numpy(pred)
    label_tensor = torch.from_numpy(label)
    predict_correct_bool = pred_tensor == label_tensor
    residual_tensor = torch.where(predict_correct_bool, 0, 1)
    return residual_tensor.cpu().numpy()


def calculate_score_residual(
        logits, labels, weights, bias, num_classes=10
    ):
    scores_dict = {}
    residuals_dict = {}
    if weights is not None:
        weight_norm = np.linalg.norm(weights, axis=1, ord=2)
    else:
        weight_norm = np.ones(num_classes)

    # === Scores used in previous version ===
    logits_tensor = torch.from_numpy(logits).to(dtype=torch.float)
    max_logit_pred = np.argmax(logits, axis=1)


    # === MaxSR 
    sr = torch.softmax(logits_tensor, dim=1)
    max_sr_scores = torch.amax(sr, dim=1).numpy()
   
    max_sr_pred = max_logit_pred
    max_sr_residuals = calculate_residual(max_sr_pred, labels)
    method_name = "max_sr"
    scores_dict[method_name] = max_sr_scores
    residuals_dict[method_name] = max_sr_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [0, "o", r"$SR_{max}$", "solid"]


    # === OURS ====
    # raw margin
    values, indices = torch.topk(logits_tensor, 2, axis=1)
    raw_margin_scores = (values[:, 0] - values[:, 1]).cpu().numpy()
    raw_margin_pred = max_logit_pred
    raw_margin_residuals = calculate_residual(raw_margin_pred, labels)
    method_name = "raw_margin"
    scores_dict[method_name] = raw_margin_scores
    residuals_dict[method_name] = raw_margin_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [4, "p", r"$RL_{conf-M}$", "solid"]
        
    # geo_margin
    geo_distance = logits / weight_norm[np.newaxis, :]
    geo_values, geo_indices = torch.topk(torch.from_numpy(geo_distance).to(dtype=torch.float), 2, axis=1)
    geo_margin_scores = (geo_values[:, 0] - geo_values[:, 1]).cpu().numpy()
    geo_margin_pred = geo_indices[:, 0].cpu().numpy()
    geo_margin_residuals = calculate_residual(geo_margin_pred, labels)
    method_name = "geo_margin"
    scores_dict[method_name] = geo_margin_scores
    residuals_dict[method_name] = geo_margin_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [3, "p", r"$RL_{geo-M}$", "solid"]


    normalized_logits_tensor = logits_tensor / torch.linalg.norm(logits_tensor, ord=2, dim=1, keepdim=True)
    # raw margin (normalized logits)
    values, indices = torch.topk(normalized_logits_tensor, 2, axis=1)
    raw_margin_scores = (values[:, 0] - values[:, 1]).cpu().numpy()
    raw_margin_pred = max_logit_pred
    raw_margin_residuals = calculate_residual(raw_margin_pred, labels)
    method_name = "raw_margin_norm"
    scores_dict[method_name] = raw_margin_scores
    residuals_dict[method_name] = raw_margin_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [2, "p", r"$RL_{norm}$", "solid"]

    return scores_dict, residuals_dict



def get_read_data_dir(model_name, dataset_name):
    abs_root = os.path.join(
        "SC-raw-data", dataset_name, model_name
    )
    return abs_root


def create_save_data_dir(model_name, dataset_name):
    save_rc_curve_dir = os.path.join("test-vis-%s" % dataset_name, "RC-Curves", model_name)
    save_confidence_root_dir = os.path.join(save_rc_curve_dir, "Conf-Histograms")
    save_rc_data_dir = os.path.join("test-vis-%s" % dataset_name, "RC-Curve-Data", model_name)
    os.makedirs(save_rc_curve_dir, exist_ok=True)
    os.makedirs(save_confidence_root_dir, exist_ok=True)
    os.makedirs(save_rc_data_dir, exist_ok=True)
    return save_rc_curve_dir, save_confidence_root_dir, save_rc_data_dir


def read_data(dir, load_classifier_weight=False):
    raw_logits = np.load(os.path.join(dir, "pred_logits.npy"))
    labels = np.load(os.path.join(dir, "labels.npy"))
    if load_classifier_weight:
        weights_dir = os.path.join(dir, "last_layer_weights.npy")
        if os.path.exists(weights_dir):
            last_layer_weights = np.load(weights_dir)
            last_layer_bias = np.load(os.path.join(dir, "last_layer_bias.npy"))
        else:
            last_layer_weights = None
            last_layer_bias = None
    else:
        last_layer_weights = None
        last_layer_bias = None
    return raw_logits, labels, last_layer_weights, last_layer_bias


def main(args):
    if args.dataset_name == "CIFAR":
        INC_CORR_TYPE_LIST = [
            "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
            "frost", "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur",
            "pixelate", "saturate", "shot_noise", "snow", "spatter",
            "speckle_noise", "zoom_blur"
        ]
        INO_LIST = [
            "cifar100"
        ]
    elif args.dataset_name == "ImageNet":
        INC_CORR_TYPE_LIST = [
            "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
            "frost", "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur",
            "pixelate", "saturate", "shot_noise", "snow", "spatter",
            "speckle_noise", "zoom_blur"
        ]
        INO_LIST = [
            "openimage-o"
        ]
    else:
        raise RuntimeError("Chech what experiment you want to do.")
    model_name = args.model_name
    read_data_root = get_read_data_dir(model_name, args.dataset_name)
    save_rc_root, save_conf_root, save_rc_data_root = create_save_data_dir(model_name, args.dataset_name)

    case_acc_dict = {}
    # === Get In-D ===
    if args.dataset_name == "CIFAR":
        in_d_data_str = "cifar10"
        in_c_data_str = "cifar10-c"
    elif args.dataset_name == "ImageNet":
        in_d_data_str = "imagenet"
        in_c_data_str = "imagenet-c"
    else:
        raise RuntimeError("Chech what experiment you want to do.")
    in_data_root = os.path.join(read_data_root, in_d_data_str)
    in_logits, in_labels, last_layer_weights, last_layer_bias = read_data(in_data_root, True)
    print("Check In-D shapes: ", in_logits.shape, in_labels.shape)
    acc = np.mean(np.argmax(in_logits, axis=1) == in_labels) * 100
    case_acc_dict["clean_val"] = acc
    # === Check Logits scale ===
    max_logits = np.amax(in_logits, axis=1)
    mean_logits, std_logits = np.mean(max_logits), np.std(max_logits)
    print("Max logit mean: %.06f | std: %.06f " % (mean_logits, std_logits))

    # Get Cov-shift Data 
    in_c_logits , in_c_labels = [], []
    for corr_type in INC_CORR_TYPE_LIST:
        for corr_level in [3]:
            # print("Read %s %s-%d data." % (in_c_data_str, corr_type, corr_level))
            in_c_path_str = "%s_%s_%d" % (in_c_data_str, corr_type, corr_level)
            in_c_data_root = os.path.join(read_data_root, in_c_path_str)
            logits, labels, _, _ = read_data(in_c_data_root)
            in_c_logits.append(logits)
            in_c_labels.append(labels)

            acc = np.mean(np.argmax(logits, axis=1) == labels) * 100
            case_acc_dict["%s" % corr_type] = acc

    in_c_logits, in_c_labels = np.concatenate(in_c_logits, axis=0), np.concatenate(in_c_labels, axis=0)
    # print("Check C shapes: ", in_c_logits.shape, in_c_labels.shape)
    # print(case_acc_dict)

    #  Get OOD data
    in_o_logits, in_o_labels = [], []
    for ood_name in INO_LIST:
        # print("Read %s data." % ood_name)
        in_o_data_root = os.path.join(read_data_root, ood_name)
        logits, labels, _, _ = read_data(in_o_data_root)
        in_o_logits.append(logits)
        in_o_labels.append(-10 * np.ones_like(labels))

    in_o_logits, in_o_labels = np.concatenate(in_o_logits, axis=0), np.concatenate(in_o_labels, axis=0)
    # print("Check OOD shapes: ", in_o_logits.shape, in_o_labels.shape)
    # === Init Dict to save RC results ===
    total_scores_dict = {}
    total_residuals_dict = {}

    # === Calculate Scores and Residuals for RC ===
    # in-d
    in_scores_dict, in_residuals_dict = calculate_score_residual(
        in_logits, in_labels,  last_layer_weights, last_layer_bias,
    )
    dataset_name = "in-d"
    total_scores_dict[dataset_name] = {}
    total_residuals_dict[dataset_name] = {}
    DATASET_NAME_LIST.append(dataset_name)
    for method_name in CF_METHOD_STR_LIST:
        total_scores_dict[dataset_name][method_name] = in_scores_dict[method_name]
        total_residuals_dict[dataset_name][method_name] = in_residuals_dict[method_name]
    
    # cov-shifted
    in_c_scores_dict, in_c_residuals_dict = calculate_score_residual(
        in_c_logits, in_c_labels, last_layer_weights, last_layer_bias,
    )
    dataset_name = "cov-shift"
    total_scores_dict[dataset_name] = {}
    total_residuals_dict[dataset_name] = {}
    DATASET_NAME_LIST.append(dataset_name)
    for method_name in CF_METHOD_STR_LIST:
        total_scores_dict[dataset_name][method_name] = in_c_scores_dict[method_name]
        total_residuals_dict[dataset_name][method_name] = in_c_residuals_dict[method_name]

    # ood
    in_o_scores_dict, in_o_residuals_dict = calculate_score_residual(
        in_o_logits, in_o_labels, last_layer_weights, last_layer_bias,
    )
    dataset_name = "ood"
    total_scores_dict[dataset_name] = {}
    total_residuals_dict[dataset_name] = {}
    DATASET_NAME_LIST.append(dataset_name)
    for method_name in CF_METHOD_STR_LIST:
        total_scores_dict[dataset_name][method_name] = in_o_scores_dict[method_name]
        total_residuals_dict[dataset_name][method_name] = in_o_residuals_dict[method_name]

    # === Plot Score Distribution ===
    method_name_list = [
        "max_sr", "raw_margin", "geo_margin", "raw_margin_norm"
    ]
    fig_name = "Clean_and_C_and_OOD.png"
    save_path = os.path.join(save_rc_root, fig_name)
    coverage_dict, residual_dict = plot_rc_curve_demo(
        total_scores_dict, total_residuals_dict, save_path, case_name="all", 
        method_name_list=method_name_list
    )

    with open(os.path.join(save_rc_data_root, "coverage.pkl"), 'wb') as fp:
        pickle.dump(coverage_dict["raw_margin"], fp)
    with open(os.path.join(save_rc_data_root, "risk.pkl"), 'wb') as fp:
        pickle.dump(residual_dict["raw_margin"], fp)
    with open(os.path.join(save_rc_data_root, "full_cvg_acc.pkl"), 'wb') as fp:
        pickle.dump(case_acc_dict, fp)

    print('RC curve data saved successfully to file')

if __name__ == "__main__":
    clear_terminal_output()
    print("This script compares the SC performance of geo/conf margins on 1) normally trained and 2) std-fc networks")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", dest="root_dir", type=str,
        default="SC-raw-data",
        help="Root dir to retrieve SC logits data."
    )
    parser.add_argument(
        "--dataset_name", dest="dataset_name",
        default="CIFAR", type=str,
    )
    args = parser.parse_args()
    root_dir = os.path.join(args.root_dir, args.dataset_name)
    model_names = [f for f in os.listdir(root_dir)]
    for model_name in model_names:
        args.model_name = model_name
        print()
        print("Examining : %s" % model_name)
        main(args)

    print("Completed.")