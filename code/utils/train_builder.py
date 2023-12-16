import torch
import os
import timm
from utils.models import ResNet34Customized, ResNet50Customized, TIMM_MODEL_CARDS, build_timm_model
from utils.loss import MarginLoss, LogitNormLoss


def get_loss(loss_config):
    msg = "  Use Loss "
    if loss_config["name"] == "CE":
        msg += "[CE]"
        loss_func = torch.nn.CrossEntropyLoss(reduction=loss_config["reduction"])
    elif loss_config["name"] == "Margin":
        msg += "[Margin] Loss | Rescale Logits: "
        rescale_logits = loss_config["rescale_logits"]
        power = loss_config["power"]
        reduction = loss_config["reduction"]
        margin=loss_config["margin"]
        temperature = loss_config["temperature"]
        loss_func = MarginLoss(
            reduction=reduction, margin=margin, p=power, 
            rescale_logits=rescale_logits,
            temperature=temperature
        )
        if rescale_logits:
            msg += "TRUE | Temperature %.04f" % temperature
        else:
            msg += "FALSE"
    elif loss_config["name"] == "LogitNorm":
        msg += "[LogitNorm] Loss | "
        reduction = loss_config["reduction"]
        temperature = loss_config["temperature"]
        loss_func = LogitNormLoss(
            t=temperature, reduction=reduction
        )
        msg += " T = %.04f " % temperature
    else:
        raise RuntimeError("Unimplemented loss type.")
    return loss_func, msg


def get_optimizer(optimizer_cfg, model):
    opt_name = optimizer_cfg["name"]
    msg = "  Use scheduler "
    if opt_name == "AdamW":
        lr = optimizer_cfg["lr"]
        weight_decay = optimizer_cfg["weight_decay"]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        msg += "[AdamW]"
    elif opt_name == "SGD":
        lr = optimizer_cfg["lr"]
        weight_decay = optimizer_cfg["weight_decay"]
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        msg += "[SGD]"
    else:
        raise RuntimeError("The author did not implement other optimizers yet.")
    return optimizer, msg


def get_scheduler(config, optimizer):
    n_epoch = config["train"]["total_epochs"]
    scheduler_name = config["train"]["scheduler"]["name"]

    msg = "  Use scheduler "
    if "Cosine" in scheduler_name:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epoch
        )
        msg += "[CosineAnnealing]"
    elif scheduler_name == "Exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config["train"]["scheduler"]["gamma"]
        )
        msg += "[Exponential]"
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config["train"]["scheduler"]["step_size"], 
            gamma=config["train"]["scheduler"]["gamma"]
        )
        msg += "[StepLR]"
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=config["train"]["scheduler"]["mode"], 
            patience=config["train"]["scheduler"]["patience"]
        )
        msg += "[ReduceLROnPlateau]"
    elif scheduler_name == "MultiStepLR":
        interval = n_epoch // 4
        milestones = [interval, interval*2, interval*3]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
    else:
        raise RuntimeError("The author did not implement other scheduler yet.")
    
    # === Add 5 warm up epochs ===
    warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm_up_scheduler, scheduler], milestones=[3]
    )
    msg += " with Linear Warm Ups (5 epochs)."
    return scheduler, msg



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

def save_model_ckp(
    model, epoch, iter_num,
    optimizer, scheduler, save_dir, name=None
    ):
    """
        Save training realted checkpoints: 
            model_state_dict, scheduler, epoch, iter, optimizer
    """
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    if name is None:
        checkpoint = {
            "epoch": epoch,
            "total_iter": iter_num,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            }
        torch.save(checkpoint, os.path.join(save_dir, "latest.pth"))
        msg = "  >> Save Training Checkpoint (for resume training purpose) at <<latest.pth>>"
    else:
        checkpoint = {
            "epoch": epoch,
            "total_iter": iter_num,
            "model_state_dict": model_state_dict,
        }
        torch.save(checkpoint, os.path.join(save_dir, "%s.pth" % name))
        msg = "  >> Save %s Model State Dict ..." % name
    return msg


def get_model(cfg):
    dataset_name = cfg["dataset"]["name"]
    num_classes = cfg["dataset"]["num_classes"]
    model_name = cfg["model"]["name"]
    standardized_fc = cfg["model"]["standardized_fc"]

    if dataset_name in ["cifar10", "cifar10-c"] and model_name in ["resnet", "resnet34"]:
        # === By default use resnet 34 ===
        model = ResNet34Customized(
            num_classes=num_classes, dim_features=512, init_weights=True, 
            standardized_linear_weights=standardized_fc
        )
    elif dataset_name in ["imagenet", "imagenet-c"]:
        if model_name in ["resnet", "resnet34"]:
            model = ResNet34Customized(
                num_classes=num_classes, dim_features=512, init_weights=True, 
                standardized_linear_weights=standardized_fc
            )
        elif model_name == "resnet50":
            model = ResNet50Customized(
                num_classes=num_classes, init_weights=True, 
                standardized_linear_weights=standardized_fc
            )
        elif model_name in TIMM_MODEL_CARDS.keys():  #  Timm Pretrained model finetuning
            model = build_timm_model(model_name, standardized_linear_weights=standardized_fc)
        else:
            raise RuntimeError("ImageNet Models unimplemented.")
    else:
        raise RuntimeError("Unsupported models.")
    msg = "  Use Model %s " % (model.__repr__)
    return model, msg