import argparse, os
from collections import namedtuple
from utils.general import load_json
from utils.dataset import get_loader_imagenet_val, get_loader_imagenet_c, get_loader_openimage_o,\
    get_loader_imagenet_o, get_loader_cifar10_val, get_loader_cifar10_c, get_loader_cifar100_val


def main(args):
    dataset_path_dict = load_json(args.dataset_path_dict)
    batch_size = args.batch_size
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



if __name__ == "__main__":
    print("Collect self-adaptive-training CIFAR models prediction logits and weight norm of the last linear layer.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", dest="dataset", type=str, default="cifar10",
        help="The dataset used to test the SC performance. [cifar10,  imagenet, cifar10-c, imagenet-c]"
    )
    parser.add_argument(
        "--model", dest="model", default="resnet", type=str,
        help="Model ARCH to collect logits."
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
        args.dataset = "cifar100"
        main(args)
        args.dataset = "cifar10"
        main(args)

    print("Completed.")