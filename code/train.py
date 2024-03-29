import sys, os, argparse, time
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
# ===
import pandas as pd
import wandb
import torch
import numpy as np
import math
from utils.general import load_json, set_random_seeds, makedir, create_log_info_file,\
    save_exp_info, set_cuda_device, print_and_log, get_optimizer_lr, check_lr_criterion, \
    save_dict_to_csv
from utils.train_builder import get_loss, get_samples, save_model_ckp, get_scheduler, \
    get_optimizer, get_model
from utils.models import ResNet34Customized, ResNet50Customized
from utils.dataset import get_loader_train


def main(cfg):
    print("===== ===== ===== ===== ")
    set_random_seeds(cfg["seed"])

    # === Init WandB Logger ===
    wandb_config = cfg["wandb"]
    if wandb_config["init"]:
        # === Use Weight and Bias to monitor experiments ===
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_config["project"],
            name=cfg["log_folder"]["save_root"]
        )

    # === Create Experiment Save Root Dir ===
    project_name = cfg["wandb"]["project"]
    exp_log_root = os.path.join(
        "..", project_name, cfg["log_folder"]["save_root"]
    )
    ckpt_dir = os.path.join(exp_log_root, "checkpoints")
    makedir(exp_log_root)
    if cfg["continue"]:
        ckpt_dir = cfg["ckpt_dir"]
    else:
        cfg["ckpt_dir"] = ckpt_dir
        makedir(ckpt_dir)
    # Create a log file & save experiment configs
    log_file = create_log_info_file(exp_log_root)
    log_mode = "a" if os.path.exists(log_file) else "w"
    csv_dir = os.path.join(exp_log_root, "log.csv")
    # Count GPU
    device, n_gpu = set_cuda_device()   # if n_gpu > 1, use data parallel

    # === Creat Model, Dataset for training and validation ===
    model, msg = get_model(cfg)
    model = model.to(device)
    print_and_log(msg, log_file, mode=log_mode)
    
    train_loader, val_loader, msg = get_loader_train(cfg["dataset"], only_val=False)
    print_and_log(msg, log_file)

    # === Create Scheduler, Optmizer 
    train_loss_func, msg = get_loss(loss_config=cfg["train"]["loss"])
    val_loss_func, _ = get_loss(loss_config=cfg["train"]["loss"])  # Use the same as training.
    print_and_log(msg, log_file)
    optimizer, msg = get_optimizer(optimizer_cfg=cfg["train"]["optimizer"], model=model, model_name=cfg["model"]["name"])
    print_and_log(msg, log_file)
    scheduler, msg = get_scheduler(cfg, optimizer=optimizer)
    print_and_log(msg, log_file)
    if cfg["continue"]:
        ckpt_file = os.path.join(ckpt_dir, "latest.pth")
        ckpt_content = torch.load(ckpt_file)
        epoch = ckpt_content["epoch"]
        total_iter = ckpt_content["total_iter"]
        model_state_dict = ckpt_content["model_state_dict"]
        opt_state_dict = ckpt_content["optimizer_state_dict"]
        scheduler_state_dict = ckpt_content["scheduler_state_dict"]
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(opt_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)
        msg = "Continue Training. Mode & Optimizer & Scheduler state dict loaded."
        print_and_log(msg, log_file)
        # Load the logged training stats history
        summary = pd.read_csv(csv_dir).to_dict("list")
        best_val_acc = np.amax(summary["val_acc"])
    else:
        epoch, total_iter = 0, 0
        # === Create Blank Logger ===
        summary = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "max_fc_weight_norm": [],
            "min_fc_weight_norm": [],
            "mean_top2_logit_diff": [],
            "std_top2_logit_diff": []
        }
        best_val_acc = 0.

    # set this flag so that we can point to the saved config file (.json) in the checkpoint folder to resumt training
    # Instead of using a new one in <config_files> folder
    cfg["continue"] = True  
    save_exp_info(exp_log_root, cfg)

    if n_gpu > 1:  # Data Parallel if use multiple GPUs
        model = torch.nn.DataParallel(model)
    
    # === Depreciated for now, cuz I see nan loss ===
    # scaler = torch.cuda.amp.GradScaler() # torch.cuda.amp.grad_scaler.GradScaler()  # For mixed precision
    
    optimizer_lr = get_optimizer_lr(optimizer)
    msg = "Initial Optimizer Learning Rate [%.6f]" % optimizer_lr
    print_and_log(msg, log_file)
    try:
        # FLAG to terminate the training process in case some scheduler turns the lr too small
        target_lr = cfg["train"]["optimizer"]["stop_lr"]
        stop_training = check_lr_criterion(optimizer_lr, target_lr)
    except:
        target_lr = 1e-12
        stop_training = False

    # For every print_loss_interval (iters), print one avg training loss to monitor the convergence
    # For every epoch, perform one validation and save ckpt, log this info into the summary dict
    rolling_train_loss_log, total_iter = [], 0
    print_loss_interval = cfg["train"]["print_interval"]
    while epoch < cfg["train"]["total_epochs"] and not stop_training:
        epoch += 1
        t_start = time.time()  # Epoch Training start time
        msg = "\n===== Training epoch [%d] =====" % epoch
        print_and_log(msg, log_file)
        optimizer_lr = get_optimizer_lr(optimizer)
        msg = "  Lr --- %.06f " % optimizer_lr
        print_and_log(msg, log_file)
        # stop_training = check_lr_criterion(optimizer_lr, target_lr)   # If lr has been too small, the final epoch to train
        
        model.train()
        for _, data in enumerate(train_loader):
            total_iter += 1
            inputs, labels = get_samples(cfg["dataset"], data, device)
            optimizer.zero_grad()

            # # == Depreciated For now: Mixed precision autocast =====
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            #     output = model(inputs)
            #     loss = train_loss_func(output, labels)
            #     if torch.isnan(loss.item()):
            #         output_norm = torch.linalg.norm(output, dim=1, ord=2, keepdim=True)
            #         msg = "  Check output norm: \n    " + output_norm.__repr__ + " \n      | Loss: {}".format(loss.item())
            #         print_and_log(msg, log_file)
            #         raise RuntimeError("Find Loss isnan.")
                
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # ========================================================

            output = model(inputs)
            loss = train_loss_func(output, labels)
            # if torch.isnan(loss.item()):
            #     output_norm = torch.linalg.norm(output, dim=1, ord=2, keepdim=True)
            #     msg = "  Check output norm: \n    " + output_norm.__repr__ + " \n      | Loss: {}".format(loss.item())
            #     print_and_log(msg, log_file)
            #     raise RuntimeError("Find Loss isnan.")
            loss.backward()
            optimizer.step()

            rolling_train_loss_log.append(loss.item())
            if total_iter % 100 == 0:
                print_and_log("   Iter [%d] - Train Loss [%.08f]" % (total_iter, loss.item()), log_file)
            # === print training loss per interval ===
            if total_iter % print_loss_interval == 0 and total_iter > 0:
                avg_loss = np.mean(rolling_train_loss_log)
                msg = "  Epoch [%d] - Total iter [%d] - AVG train loss - [%.08f]" % (
                    epoch, total_iter, avg_loss
                )
                print_and_log(msg, log_file)

        t_end = time.time()  # Epoch Training end time
        msg = "  Epoch [%d] training time - %04f " % (epoch, t_end-t_start)
        print_and_log(msg, log_file)
        
        avg_train_loss_epoch = np.mean(rolling_train_loss_log)
        rolling_train_loss_log = []
        summary["train_loss"].append(avg_train_loss_epoch)
        msg = "  Epoch [%d] avg training loss - [%.08f]" % (epoch, avg_train_loss_epoch)
        print_and_log(msg, log_file)

        # === save ckpt for continue training === 
        save_model_ckp(
            model, epoch, total_iter, optimizer, scheduler, ckpt_dir
        )

        msg = "===== Validation Epoch [%d] ====" % epoch
        print_and_log(msg, log_file)

        # === Check fc layer weights and bias norm ===
        if isinstance(model, ResNet34Customized):
            if n_gpu > 1:
                last_layer = model.module.classifier[-1]
            else:
                last_layer = model.classifier[-1]
        elif isinstance(model, ResNet50Customized):
            if n_gpu > 1:
                last_layer = model.module.features.fc
            else:
                last_layer = model.features.fc
        elif "dino" in cfg["model"]["name"]:
            if n_gpu > 1:
                last_layer = model.module.linear_head
            else:
                last_layer = model.linear_head
        else:
            raise RuntimeError("Unimplemented weight norm.")
        weights = last_layer.weight.data.clone().cpu().numpy()
        bias = last_layer.bias.data.clone().cpu().numpy()[:, np.newaxis]
        vec_aug = np.concatenate([weights, bias], axis=1)
        weight_norm = np.linalg.norm(weights, ord=2, axis=1)
        aug_weight_norm = np.linalg.norm(vec_aug, ord=2, axis=1)
        w_min, w_max = np.amin(weight_norm), np.amax(weight_norm)
        w_aug_min, w_aug_max = np.amin(aug_weight_norm), np.amax(aug_weight_norm)
        msg = " Check last layer weight norms: \n"
        msg += "    Last later weights (excl. bias) --- Min %.06f | Max %.06f  \n" % (w_min, w_max)
        msg += "    Last later weights (incl. bias) --- Min %.06f | Max %.06f" % (w_aug_min, w_aug_max)
        print_and_log(msg, log_file)

        validation_loss_log = []
        val_correct, val_total_samples = 0., 0
        model.eval()
        top2_logit_diff_log = []
        with torch.no_grad():
            for _, data in enumerate(val_loader):
                inputs, labels = get_samples(cfg["dataset"], data, device)
                output = model(inputs)
                loss = val_loss_func(output, labels)
                validation_loss_log.append(loss.item())

                pred = output.argmax(1)
                val_correct += (pred == labels).sum().item()
                val_total_samples += labels.shape[0]

                top2_logits, _ = torch.topk(output, k=2, dim=1)
                top2_logit_diff = (top2_logits[:, 0] - top2_logits[:, 1]).cpu().numpy()
                top2_logit_diff_log.append(top2_logit_diff)

        val_acc = val_correct / (val_total_samples) * 100
        val_loss_epoch = np.mean(validation_loss_log)
        msg = "  Epoch [%d] Validation loss - [%.08f] | Validation acc - [%.04f %%]" % (
            epoch, val_loss_epoch, val_acc
        )
        print_and_log(msg, log_file)

        top2_logit_diff_log = np.concatenate(top2_logit_diff_log, axis=0)
        mean_top2_logit_diff, std_top2_logit_diff = np.mean(top2_logit_diff_log), np.std(top2_logit_diff_log)
        msg = " Check Top2 Logit Diff Stats. (val set) \n"
        msg += "     ---- Mean %.06f | Std %.06f " % (mean_top2_logit_diff, std_top2_logit_diff)
        print_and_log(msg, log_file)

        # === Log Training Stats ===
        summary["val_loss"].append(val_loss_epoch)
        summary["val_acc"].append(val_acc)
        summary["max_fc_weight_norm"].append(w_aug_max)
        summary["min_fc_weight_norm"].append(w_aug_min)
        summary["mean_top2_logit_diff"].append(mean_top2_logit_diff)
        summary["std_top2_logit_diff"].append(std_top2_logit_diff)
        save_dict_to_csv(summary, csv_dir)
        if wandb_config["init"]:
            wandb.log({
                "train_loss": avg_train_loss_epoch, 
                "val_loss": val_loss_epoch,
                "val_acc": val_acc,
                "lr": optimizer_lr,
                "max_fc_weight_norm": w_aug_max,
                "min_fc_weight_norm": w_aug_min,
                "mean_top2_logit_diff": mean_top2_logit_diff,
            })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_ckpt_name = "best"
            save_model_ckp(
                model, epoch, total_iter, optimizer, scheduler, ckpt_dir, name=best_model_ckpt_name
            )
            msg = "Best Model achieved so far at Epoch [%d] with Val Acc [%.08f]" % (epoch, val_acc)
            print_and_log(msg, log_file)
        
        # === Scheduler Update ===
        if cfg["train"]["scheduler"]["name"] != "ReduceLROnPlateau":
            scheduler.step()
        else:
            if cfg["train"]["scheduler"]["mode"] == "min":  # Reduce on loss plateau
                scheduler.step(avg_train_loss_epoch)
            else:
                scheduler.step(val_acc)  # Reduce on val acc plateau



if __name__ == "__main__":
    print("This is the trainig script of sc (margin promoted) training.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_files', 'a_working_config_example.json'),
        help="Path to the json config file for training."
    )
    args = parser.parse_args()
    cfg = load_json(args.config)
    main(cfg)
    print("All tasks Completed.")