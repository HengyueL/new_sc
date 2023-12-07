import os, json, random
import torch
import numpy as np


def load_json(json_path):
    content = json.load(open(json_path, "r"))
    return content


def set_random_seeds(seed):
    """
        This function sets all random seed used in this experiment.
        For reproduce purpose.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def makedir(dir):
    os.makedirs(dir, exist_ok=True)


def makedirs(dir_list):
    for dir in dir_list:
        makedir(dir)


def create_log_info_file(dir, name="experiment_log.txt"):
    log_file = os.path.join(dir, name)
    return log_file


def save_dict_to_json(dict, save_dir):
    with open(save_dir, "w") as outfile:
        json.dump(dict, outfile, indent=4)


def save_exp_info(save_dir, config):
    """
        Create new log folder / Reuse the checkpoint folder for experiment.
    """
    # Save Exp Settings as Json File
    exp_config_file = os.path.join(save_dir, "Exp_Config.json")
    save_dict_to_json(config, exp_config_file)


def count_cuda_devices():
    """
        This function returns a list of available gpu devices.

        If GPUs do noe exist, return: None
    """
    n_gpu = torch.cuda.device_count()
    return n_gpu


def set_cuda_device():
    """
        This function set the default device to move torch tensors.

        If GPU is availble, default_device = torch.device("cuda:0").
        If GPU is not available, default_device = torch.device("cpu").
    """
    n_gpu = count_cuda_devices()
    if n_gpu < 1:
        return torch.device("cpu"), n_gpu
    else:
        return torch.device("cuda"), n_gpu


def write_log_txt(file_name, msg, mode="a"):
    """
        Write training msg to file in case of cmd print failure in MSI system.
    """
    with open(file_name, mode) as f:
        f.write(msg)
        f.write("\n")


def print_and_log(msg, log_file_name, mode="a", terminal_print=True):
    """
        Write msg to a text file.
    """
    if type(msg) == str:
        if terminal_print == True:
            print(msg)
        write_log_txt(log_file_name, msg, mode=mode)
    elif type(msg) == list:
        for word in msg:
            print_and_log(word, log_file_name, mode=mode, terminal_print=terminal_print)
    else:
        assert RuntimeError("msg input only supports string / List input.")


def get_optimizer_lr(optimizer):
    for param_group in optimizer.param_groups:
        optimizer_lr = param_group["lr"]
    return optimizer_lr



def check_lr_criterion(lr, target_lr):
    """
        True if meet lr criterion
    """
    return lr <= target_lr