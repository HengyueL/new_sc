import torch


def get_loss(loss_config):
    if loss_config["name"] == "CE":
        loss_func = torch.nn.CrossEntropyLoss(reduction=loss_config["reduction"])
    else:
        raise RuntimeError("Unimplemented loss type.")
    return loss_func


def get_optimizer(optimizer_cfg, model):
    opt_name = optimizer_cfg["type"]

    if opt_name == "AdamW":
        lr = optimizer_cfg["lr"]
        weight_decay = optimizer_cfg["weight_decay"]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif opt_name == "SGD":
        lr = optimizer_cfg["lr"]
        weight_decay = optimizer_cfg["weight_decay"]
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise RuntimeError("The author did not implement other optimizers yet.")
    return optimizer


def get_samples(data_config, data_from_loader, device):
    if "imagenet" in data_config["name"]:
        inputs = data_from_loader[0].to(device)
        labels = data_from_loader[1].to(device)
    elif "cifar" in data_config["name"]:
        inputs = data_from_loader[0].to(device)
        labels = data_from_loader[1].to(device)
    else:
        raise RuntimeError("Unsupported Dataset")
    return inputs, labels