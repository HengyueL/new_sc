import argparse, os
from collections import namedtuple
from utils.dataset import get_loader_imagenet_val, get_loader_imagenet_c, get_loader_openimage_o,\
    get_loader_imagenet_o


def main(args):
    # === Construct Dataset and Paths ===
    if args.dataset == 'imagenet':
        data_path = "/scratch.global/liang656/ImageNet2012"
        test_loader = get_loader_clean(
            data_path
        )
    elif args.dataset == 'imagenet-c':
        data_path = "/scratch.global/liang656/ImageNet-C/"
        test_loader = get_loader_c(data_path, args.corr_type, args.corr_level)
    elif args.dataset == "imagenet-o":
        data_path = "/scratch.global/liang656/ImageNet-O"
        test_loader = get_loader_o(
            data_path
        )
    elif args.dataset == "openimage-o":
        data_path = "/scratch.global/liang656/OpenImage"
        txt_path = "/home/jusun/liang656/SelectiveCls/code/openimage_o.txt"
        test_loader = get_loader_openimage_o(
            data_path, txt_path
        )
    elif args.dataset == 'cifar10':
        data_path = "/home/jusun/liang656/datasets/Cifar10/Clean"
        test_set = CIFAR10(
            root=data_path, train=False, transform=tform_cifar, download=False
        )
        test_loader = DataLoader(
            test_set, batch_size=256, shuffle=False, num_workers=8
        )
    elif args.dataset == 'cifar10-c':
        print("Cifar-10-c")
        data_path = "/home/jusun/liang656/datasets/Cifar10/CIFAR10-C"
        test_set = CifarCDataset(
            root_dir=data_path, corr_type=args.corr_type,
            corr_level=args.corr_level, transform=tform_cifar
        )    
        test_loader = DataLoader(
            test_set, batch_size=256, shuffle=False, num_workers=8
        )
    elif args.dataset == "cifar100":
        print("CIFAR100")
        data_path = "/home/jusun/liang656/datasets/Cifar100/Clean"
        test_set = CIFAR100(
            root=data_path, train=False, transform=tform_cifar, download=True
        )
        test_loader = DataLoader(
            test_set, batch_size=256, shuffle=False, num_workers=8
        )
    else:
        raise RuntimeError("Undefined dataset.")



if __name__ == "__main__":
    print("Collect self-adaptive-training CIFAR models prediction logits and weight norm of the last linear layer.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", dest="dataset", type=str,
        default="cifar10",
        help="The dataset used to test the SC performance. [cifar10,  imagenet, cifar10-c, imagenet-c]"
    )
    parser.add_argument(
        "--model", dest="model",
        default="resnet",
        help="Model ARCH to collect logits."
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