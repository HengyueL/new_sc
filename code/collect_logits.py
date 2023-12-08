import argparse, os
import torch
import numpy as np
from utils.train_builder import get_model
from utils.general import load_json
from utils.dataset import get_loader_imagenet_val, get_loader_imagenet_c, get_loader_openimage_o,\
    get_loader_imagenet_o, get_loader_cifar10_val, get_loader_cifar10_c, get_loader_cifar100_val


def main(args):
    dataset_path_dict = load_json(args.dataset_path_dict)
    batch_size = args.batch_size
    device = torch.device("cuda")
    # === Construct Dataset and Paths ===
    if args.dataset == 'imagenet':
        print("imagenet")
        data_path = dataset_path_dict[args.dataset]
        test_loader = get_loader_imagenet_val(
            data_path, batch_size=batch_size
        )
    elif args.dataset == 'imagenet-c':
        print("imagenet-c - %s - %d" % (args.corr_type, args.corr_level) )
        data_path = dataset_path_dict[args.dataset]
        test_loader = get_loader_imagenet_c(
            data_path, args.corr_type, args.corr_level, batch_size=batch_size
        )
    elif args.dataset == "imagenet-o":
        print("imagenet-o")
        data_path = dataset_path_dict[args.dataset]
        test_loader = get_loader_imagenet_o(
            data_path, batch_size
        )
    elif args.dataset == "openimage-o":
        print("openimage-o")
        data_path = dataset_path_dict[args.dataset]["data_path"]
        txt_path = dataset_path_dict[args.dataset]["annot_file_path"]
        test_loader = get_loader_openimage_o(
            data_path, txt_path, batch_size
        )
    elif args.dataset == 'cifar10':
        print("CIFAR10")
        data_path = dataset_path_dict[args.dataset]
        test_loader = get_loader_cifar10_val(
            data_path, batch_size
        )
    elif args.dataset == 'cifar10-c':
        print("Cifar-10-c")
        data_path = dataset_path_dict[args.dataset]
        test_loader = get_loader_cifar10_c(
            data_path, args.corr_type, args.corr_level, batch_size=batch_size
        )
    elif args.dataset == "cifar100":
        print("CIFAR100")
        data_path = "/home/jusun/liang656/datasets/Cifar100/Clean"
        test_loader = get_loader_cifar100_val(
            data_path, batch_size
        )
    else:
        raise RuntimeError("Undefined dataset.")
    
    # === Generate Model config (in the training format so that we can reuse the function wrapper)
    training_config = load_json(args.model_config)
    model = get_model(training_config)
    ckpt_dir = os.path.join(
        training_config["ckpt_dir"], "best.pth"
    )
    model_state_dict = torch.load(ckpt_dir)["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    print("Pretrained Model Loaded")

    model_id_str = training_config["log_folder"]["save_root"]
    use_std_fc_weights = training_config["model"]["standardized_fc"]
    
    # === Create Standardized save dir ===
    dataset_str = args.dataset
    if dataset_str == "cifar10-c":
        corr_str = args.corr_type
        level_str = "%d" % args.corr_level
        name_str = "%s_%s_%s" % (dataset_str, corr_str, level_str) 
        save_data_root = os.path.join("..", "SC-raw-data", model_id_str, "CIFAR", name_str)
    elif dataset_str in ["cifar10", "cifar100", "imagenet", "imagenet-o, openimage-o"]:
        if "cifar" in dataset_str:
            save_data_root = os.path.join("..", "SC-raw-data", model_id_str, "CIFAR", dataset_str)
        else:
            save_data_root = os.path.join("..", "SC-raw-data", model_id_str, "ImageNet", dataset_str)
    elif dataset_str == "imagenet-c":
        corr_str = args.corr_type
        level_str = "%d" % args.corr_level
        name_str = "%s_%s_%s" % (dataset_str, corr_str, level_str) 
        save_data_root = os.path.join("..", "cifar100", model_id_str, "ImageNet", name_str)
    else:
        raise RuntimeError("UNsupported Dataset.")
    os.makedirs(save_data_root, exist_ok=True)
    
    # === Loop and get labels and pred_logits ===
    logits_log = []
    label_log = []
    # features_log = []
    with torch.no_grad():
        for _, (input, target) in enumerate(test_loader):
            input = input.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.long)
            # compute output
            logit_output = model(input)

            # == Construct logged stats ==
            logits = logit_output.cpu().numpy()
            if args.dataset in ["cifar100", "imagenet-o", "openimage-o"]:
                labels = -100 * torch.ones_like(target).cpu().numpy()
            else:
                labels = target.cpu().numpy()
            # features = model.features(input).view(input.size(0), -1).cpu().numpy()

            logits_log.append(logits)
            label_log.append(labels)
            # features_log.append(features)
    print("Outputs successfully collected. Now save data. ")
    save_logits_name = os.path.join(save_data_root, "pred_logits.npy")
    np.save(save_logits_name, np.concatenate(logits_log, axis=0))
    save_labels_name = os.path.join(save_data_root, "labels.npy")
    np.save(save_labels_name, np.concatenate(label_log, axis=0))
    # save_features_name = os.path.join(save_data_root, "features.npy")
    # np.save(save_features_name, np.concatenate(features_log, axis=0))

    print("Save final classifier (fc) weight and bias if not use standardized fc layer.")
    # ====
    if not use_std_fc_weights:
        save_weight_name = os.path.join(
            save_data_root, "last_layer_weights.npy"
        )
        save_bias_name = os.path.join(
            save_data_root, "last_layer_bias.npy"
        )

        last_layer = model.classifier[-1]
        weights = last_layer.weight.data.clone().cpu().numpy()
        bias = last_layer.bias.data.clone().cpu().numpy()

        np.save(save_weight_name, weights)
        np.save(save_bias_name, bias)
    
    print(
        "Final shape Check: ", 
        np.concatenate(logits_log, axis=0).shape,
        np.concatenate(label_log, axis=0).shape
    )

    if "-c" not in args.dataset and "100" not in args.dataset and "-o" not in args.dataset:
        logits_np = np.concatenate(logits_log, axis=0)
        labels_np = np.concatenate(label_log, axis=0)
        acc = np.mean(np.argmax(logits_np, axis=1) == labels_np) * 100
        print("%s Clean Acc: %.04f" % (args.dataset, acc))


if __name__ == "__main__":
    print("Collect self-adaptive-training CIFAR models prediction logits and weight norm of the last linear layer.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", dest="dataset", type=str, default="cifar10",
        help="The dataset used to test the SC performance. [cifar10,  imagenet, cifar10-c, imagenet-c]"
    )
    parser.add_argument(
        "--model_config", dest="model_config", default="", type=str,
        help="Saved configuration file (used to regenerate the model)."
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", default=256, type=int,
        help="Batch size of the validation experiment."
    )
    parser.add_argument(
        "--dataset_path_dict", dest="dataset_path_dict", default="config_files/dataset_path.json", type=str,
        help="A .json file that stores all the dataset paths to avoid constant coding."
    )
    args = parser.parse_args()

    if "imagenet" in args.dataset:
        if args.dataset == "imagenet-c":
            root_dir = "/scratch.global/liang656/ImageNet-C/"
            corr_type_list = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
            # corr_level_list = list(range(1, 6, 1))
            corr_level_list = [3]
            for corr_type in corr_type_list:
                for corr_level in corr_level_list:
                    args.corr_type = corr_type
                    args.corr_level = corr_level
                    main(args)
        args.dataset = "imagenet"
        main(args)
        args.dataset = "imagenet-o"
        main(args)
        args.dataset = "openimage-o"
        main(args)
    elif "cifar" in args.dataset:
        if args.dataset == "cifar10-c":
            root_dir = "/home/jusun/liang656/datasets/Cifar10/CIFAR10-C"
            corr_type_list = [f.split(".npy")[0] for f in os.listdir(root_dir) if "labels" not in f]
            # corr_level_list = list(range(1, 6, 1))
            corr_level_list = [3]
            for corr_type in corr_type_list:
                for corr_level in corr_level_list:
                    args.corr_type = corr_type
                    args.corr_level = corr_level
                    main(args)
        args.dataset = "cifar10"
        main(args)
        args.dataset = "cifar100"
        main(args)

    print("Completed.")