
import os
from collections import namedtuple
import torchvision
from torchvision.datasets import CIFAR10, ImageNet, CIFAR100
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image


DC = namedtuple(
    'DatasetConfig', ['mean', 'std', 'input_size', 'num_classes']
)
DATASET_CONFIG = {
    'cifar10': DC([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616], 32, 10),
    'cifar10-c': DC([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616], 32, 10),
    'cifar100': DC([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616], 32, 100),
    'imagenet': DC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224, 1000),
    'imagenet-c': DC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224, 1000),
    'imagenet-o': DC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224, 1000),
    'openimage-o': DC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224, 1000),
} 


def get_trainsform(name, normalize=True, is_train=False):
    transform = []

    # size transform specific for dataset
    if "imagenet" in name:
        transform.extend([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
        ])

    # augmentations specific for training
    if is_train:
        if name == "cifar10":
            transform.extend([
                torchvision.transforms.RandomHorizontalFlip(),
            ])
        elif name == "imagenet":
            transform.extend([
                torchvision.transforms.RandomHorizontalFlip(),
            ])
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
    

# === Get dataset and loaders  (for training purpose with config files input) ===
def get_dataset(dataset_cfg):
    dataset_name = dataset_cfg["name"]
    dataset_dir = dataset_cfg["dataset_path"]
    train_transform = get_trainsform(dataset_name, is_train=True)
    val_transform = get_trainsform(dataset_name, is_train=False)
    if dataset_name == "cifar10":
        train_set = CIFAR10(
            root=dataset_dir, train=True, transform=train_transform, download=False
        )
        val_set = CIFAR10(
            root=dataset_dir, train=False, transform=val_transform, download=False
        )
    elif dataset_name == "imagenet":
        train_set = ImageNet(
            dataset_dir, split="train", transform=train_transform
        )
        val_set = ImageNet(
            dataset_dir, split="val", transform=val_transform
        )
    else:
        raise RuntimeError("Unrecognized dataset.")

    return train_set, val_set


def get_loader_train(dataset_cfg, only_val=True):
    train_set, val_set = get_dataset(dataset_cfg)
    batch_size = dataset_cfg["batch_size"]
    num_workers = dataset_cfg["num_workers"]
    msg = "  Building dataset - [%s]" % dataset_cfg["name"]
    if only_val:
        train_loader = None
    else:
        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    val_loader = DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, msg


# === Dataset and Loaders used in the SC testing phase ===
class CifarCDataset(Dataset):
    """
        CIFAR10-C dataset.
    """
    def __init__(self, root_dir, corr_type,
                 corr_level,
                 transform=None):
        super(CifarCDataset).__init__()
        corr_data_file = os.path.join(root_dir, corr_type.lower() + ".npy")
        label_file = os.path.join(root_dir, "labels.npy")

        start_idx = (corr_level-1)*10000
        end_idx = (corr_level)*10000

        self.data = np.load(corr_data_file)[start_idx:end_idx, :, :, :]
        self.data = self.data / 255.
        self.labels = np.load(label_file)[start_idx:end_idx]
        self.transfrom = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx, :, :, :]
        label = self.labels[idx].astype(int)

        if self.transfrom is not None:
            data = self.transfrom(data)

        return data, label


# CIFAR10 Clean validation loader
def get_loader_cifar10_val(
    data_path, batch_size=512
):
    val_transform = get_trainsform(name="cifar10", is_train=False)
    val_dataset = CIFAR10(
        data_path, train=False, transform=val_transform, download=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    return val_loader


# CIFAR10-C loader
def get_loader_cifar10_c(
    data_path, corr_type, corr_level, batch_size=512
):
    val_transform = get_trainsform(name="cifar10-c", is_train=False)
    val_dataset = CifarCDataset(
        data_path, corr_type, corr_level, transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    return val_loader


# CIFAR100 val loader
def get_loader_cifar100_val(
    data_path, batch_size=512
):
    val_transform = get_trainsform(name="cifar100", is_train=False)
    val_dataset = CIFAR100(
        data_path, train=False, transform=val_transform, download=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    return val_loader


# ImageNet (Clean) validation set
def get_loader_imagenet_val(
    data_path, batch_size=512
):
    val_transform = get_trainsform(name="imagenet", is_train=False)
    dataset_val = ImageNet(
        data_path, split="val", transform=val_transform
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False
    )
    return val_loader

# ImageNet-C validation set
def get_loader_imagenet_c(
    data_path, corr_type, corr_level, batch_size=512
):
    data_path = os.path.join(
        data_path, corr_type, str(corr_level)
    )
    val_transform = get_trainsform(name="imagenet-c", is_train=False)
    dataset_val = ImageNet(
        data_path, split="val", transform=val_transform
    )
    val_loader = DataLoader(
        dataset=dataset_val, batch_size=batch_size, shuffle=False
    )
    return val_loader


# ImageNet-O validation set
def get_loader_imagenet_o(data_path, batch_size=512):
    val_transforms = get_trainsform(name="imagenet-o", is_train=False)
    dataset = ImageFolder(
        root=data_path, transform=val_transforms
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    return val_loader


# OpenImage-O dataset and loader
def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """flist format: impath label\nimpath label\n."""
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            data = line.strip().rsplit(maxsplit=1)
            if len(data) == 2:
                impath, imlabel = data, 
            else:
                impath, imlabel = data[0], -10
            imlist.append((impath, int(imlabel)))

    return imlist


class OpenImageDataset(Dataset):
    def __init__(
        self,
        root,
        flist,
        transform=None,
        target_transform=None,
        flist_reader=default_flist_reader,
        loader=default_loader
    ):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def get_loader_openimage_o(data_path, text_path, batch_size=512):
    val_transforms = get_trainsform(name="openimage-o", is_train=False)
    dataset_path = data_path
    annot_txt_path = text_path
    dataset = OpenImageDataset(
        dataset_path, annot_txt_path, transform=val_transforms
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    return val_loader