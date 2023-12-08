
from collections import namedtuple
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

DC = namedtuple(
    'DatasetConfig', ['mean', 'std', 'input_size', 'num_classes']
)
DATASET_CONFIG = {
    'cifar10': DC([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784], 32, 10),
    'imagenet': None
} 


def get_trainsform(name, input_size=None, normalize=True, is_train=False):
    transform = []

    # arugmentation
    if is_train:
        if name == "cifar10":
            transform.extend([
                torchvision.transforms.RandomHorizontalFlip(),
            ])
        elif name == "imagenet":
            raise RuntimeError("Not impletented Imagenet Transform Yet")
        else:
            raise RuntimeError("Un recognized dataset")

    # to tensor
    transform.extend([torchvision.transforms.ToTensor(),])

    # normalize
    if normalize:
        transform.extend([
            torchvision.transforms.Normalize(
                mean=DATASET_CONFIG[name].mean, 
                std=DATASET_CONFIG[name].std
            ),
        ])
    return torchvision.transforms.Compose(transform)
    

def get_dataset(dataset_cfg):
    dataset_name = dataset_cfg["name"]
    dataset_dir = dataset_cfg["dataset_path"]
    
    if dataset_name == "cifar10":
        train_transform = get_trainsform(dataset_name, is_train=True)
        train_set = CIFAR10(
            root=dataset_dir, train=True, transform=train_transform, download=False
        )
        val_transform = get_trainsform(dataset_name, is_train=False)
        val_set = CIFAR10(
            root=dataset_dir, train=False, transform=val_transform, download=False
        )
    elif dataset_name == "imagenet":
        raise RuntimeError("ImageNet dataset unimplemetend.")
    else:
        raise RuntimeError("Unrecognized dataset.")

    return train_set, val_set


def get_loader(dataset_cfg, only_val=True):
    train_set, val_set = get_dataset(dataset_cfg)
    batch_size = dataset_cfg["batch_size"]
    num_workers = dataset_cfg["num_workers"]
    msg = "  Building dataset - [%s]" % dataset_cfg["name"]
    if only_val:
        train_loader = None
    else:
        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
    val_loader = DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, msg