import sys, os, argparse, time
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
# ===
import pandas as pd
import wandb
import torch
import numpy as np
from utils.general import load_json, set_random_seeds, makedir, create_log_info_file,\
    save_exp_info, set_cuda_device, print_and_log, get_optimizer_lr, check_lr_criterion
from utils.train_builder import get_loss, get_samples



def main(cfg):
    set_random_seeds(cfg["seed"])

    # === Init WandB Logger ===
    wandb_config = cfg["wandb"]
    wandb.init(
        # set the wandb project where this run will be logged
        project=wandb_config["project"],
        # track hyperparameters and run metadata
        # config={
        #     "learning_rate": 0.02,
        #     "architecture": "CNN",
        #     "dataset": "CIFAR-100",
        #     "epochs": 10,
        # }
    )

    # === Create Experiment Save Root Dir ===
    exp_log_root = os.path.join(
        "..", "log_folder", cfg["log_folder"]["save_root"]
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
    csv_dir = os.path.join(exp_log_root, "log.csv")
    # Count GPU
    device, n_gpu = set_cuda_device()   # if n_gpu > 1, use data parallel

    
    # === Creat Model, Dataset for training and validation ===
    model, msg = build_model(config=cfg)
    print_and_log(msg, log_file, mode="w")
    
    train_loader, val_loader, msg = build_dataloader(config=cfg)
    print_and_log(msg, log_file)

    # === Create Scheduler, Optmizer 
    train_loss_func = get_loss(loss_config=cfg["train"]["loss"])
    optimizer = get_optimizer(optimizer_cfg=cfg["train"]["optimizer"], model=model)
    scheduler = get_scheduler(schduler_cfg["train"]["scheduler"], optimizer=optimizer)
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
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
    # set this flag so that we can point to the saved config file (.json) in the checkpoint folder to resumt training
    # Instead of using a new one in <config_files> folder
    cfg["resume"] = True  
    save_exp_info(exp_log_root, cfg)

    if n_gpu > 1:  # Data Parallel if use multiple GPUs
        model = torch.nn.DataParallel(model)
    scaler = torch.cuda.amp.GradScaler() # torch.cuda.amp.grad_scaler.GradScaler()  # For mixed precision
    
    optimizer_lr = get_optimizer_lr(optimizer)
    msg = "Initial Optimizer Learning Rate [%.6f]" % optimizer_lr
    print_and_log(msg, log_file)
    try:
        # FLAG to terminate the training process in case some scheduler turns the lr too small
        target_lr = cfg["optimizer"]["stop_lr"]
        stop_training = check_lr_criterion(optimizer_lr, target_lr)
    except:
        stop_training = False

    # === For every several hundred iters, log one avg training loss to monitor the convergence
    # For every epoch, perform one validation and save ckpt
    rolling_train_loss_log, total_iter = [], 0
    while epoch < cfg["train"]["total_epochs"] and not stop_training:
        epoch += 1
        t_start = time.time()  # Epoch Training start time
        msg = "===== Training epoch [%d] ====="
        print_and_log(msg, log_file)
        for batch_idx, data in enumerate(train_loader):
            total_iter += 1
            inputs, labels = get_samples(cfg["dataset"], data, device)
            optimizer.zero_grad()

            # Mixed precision autocast
            with torch.autocast(device_type="cuda"):
                output = model(inputs)
                loss = train_loss_func(output, labels)
            loss.backward()
            optimizer.step()

            # === Log Training Loss ===

        t_end = time.time()  # Epoch Training end time
        msg = "  Epoch [%d] training time - %04f " % (epoch, t_end-t_start)
        print_and_log(msg, log_file)


if __name__ == "__main__":
    print("This is the trainig script of sc (margin promoted) training.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_files', 'train_normalized_cls.json'),
        help="Path to the json config file for training."
    )
    args = parser.parse_args()
    cfg = load_json(args.config)
    main(cfg)
    print("All tasks Completed.")