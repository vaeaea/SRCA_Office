import argparse
import copy
import datetime
import json
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from thop import profile
# 统计flop、gpu、时长
from torch.cuda.amp import GradScaler
from torchinfo import summary
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE, MAE
from lib.data_prepare import get_dataloaders_from_index_data
from model.SRCA import SRCA
sys.path.append("..")


available_models = {
    "SRCA": SRCA,
}


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    batch_loss_mae_list = []

    inference_start_time = time.time()
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        if "SRCA" in type(model).__name__:
            out_batch, loss_route = model(x_batch)
        else:
            out_batch = model(x_batch)
            loss_route = torch.tensor(0.0).to(DEVICE)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item() + loss_route.item())
        batch_loss_mae_list.append(MaskedMAELoss()(out_batch, y_batch).item())

    # 计算推理时间
    inference_time = time.time() - inference_start_time

    return np.mean(batch_loss_list), np.mean(batch_loss_mae_list), inference_time


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        if "SRCA" in type(model).__name__:
            out_batch, loss_route = model(x_batch)
        else:
            out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
        model, trainset_loader, valset_loader, optimizer, scheduler, scaler, criterion, clip_grad, log=None, epoch=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    batch_loss_mae_list = []
    epoch_start_time = time.time()

    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        # print(type(model).__name__)
        if "SRCA" in type(model).__name__:
            out_batch, loss_route = model(x_batch)
        else:
            out_batch = model(x_batch)
            loss_route = torch.tensor(0.0).to(DEVICE)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch) + loss_route * 10
        batch_loss_list.append(loss.item())
        batch_loss_mae_list.append(MaskedMAELoss()(out_batch, y_batch).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_time = time.time() - epoch_start_time
    epoch_loss = np.mean(batch_loss_list)
    epoch_mae_loss = np.mean(batch_loss_mae_list)

    if "SRCA" in type(model).__name__ and cfg['optimizer'] != 'multistep':
        val_mae = MAE(*predict(model, valset_loader))
        scheduler.step(val_mae)
    else:
        scheduler.step()
    return epoch_loss, epoch_mae_loss, epoch_time


def calculate_flops(model, sample_input):
    """
    计算模型的FLOPs
    """
    model.eval()
    # 确保输入和模型在同一设备上
    if sample_input.device != next(model.parameters()).device:
        sample_input = sample_input.to(next(model.parameters()).device)

    with torch.no_grad():
        flops, params = profile(model, inputs=(sample_input,), verbose=False)
    return flops, params


def get_gpu_memory_usage():
    """
    获取当前GPU内存使用情况
    """
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        return gpu_memory_allocated, gpu_memory_reserved
    return 0, 0


def train(
        model,
        trainset_loader,
        valset_loader,
        testset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=0,
        max_epochs=200,
        early_stop=10,
        verbose=1,
        plot=False,
        log=None,
        save=None,
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []
    test_loss_list = []

    # 记录每个epoch的时间
    epoch_times = []

    # 记录GPU内存使用情况
    gpu_memory_stats = []
    scaler = GradScaler()

    for epoch in range(max_epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
            gpu_memory_before_epoch = torch.cuda.memory_allocated() / 1024 ** 3  # GB

        train_loss, train_loss_, epoch_time = train_one_epoch(
            model, trainset_loader, valset_loader, optimizer, scheduler, scaler, criterion, clip_grad, log=log,
            epoch=epoch
        )
        train_loss_list.append(train_loss)
        epoch_times.append(epoch_time)

        if torch.cuda.is_available():
            gpu_memory_after_train = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            gpu_max_memory_during_train = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB

        val_loss, val_loss_, val_inference_time = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        test_loss, test_loss_, test_inference_time = eval_model(model, testset_loader, criterion)
        test_loss_list.append(test_loss)

        y_true, y_pred = predict(model, testset_loader)

        rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
        LR = optimizer.param_groups[0]['lr']

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                f" \tLR = {LR}",

                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                "Test Loss = %.5f" % test_loss,

                " \tTrain Mae Loss = %.5f" % train_loss_,
                "Val Mae Loss = %.5f" % val_loss_,
                "Test Mae Loss = %.5f" % test_loss_,

                " \tTest Mae Loss = %.5f" % mae_all,
                "Test RMSE Loss = %.5f" % rmse_all,
                "Test MAPE Loss = %.5f" % mape_all,

                " \tEpoch Time = %.2fs" % epoch_time,
                "Val Inference Time = %.2fs" % val_inference_time,

                " \tGPU Memory before epoch = %.2fs" % gpu_memory_before_epoch,
                "GPU Max Memory during epoch = %.2fs" % gpu_max_memory_during_train,
                "GPU Memory after epoch = %.2fs" % gpu_memory_after_train,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait > 20:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch + 1}\n"
    out_str += f"Best at epoch {best_epoch + 1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )

    # 添加统计信息
    avg_epoch_time = np.mean(epoch_times)
    total_training_time = np.sum(epoch_times)
    out_str += f"Avg Epoch Time = %.2fs\n" % avg_epoch_time
    out_str += f"Total Training Time = %.2fs\n" % total_training_time

    print_log(out_str, log=log)

    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(0, epoch + 1), epoch_times, "-")
        plt.title("Epoch Time")
        plt.xlabel("Epoch")
        plt.ylabel("Time (s)")
        plt.tight_layout()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument('--mode', type=str, default='mean', choices=['mean', 'add'], help='Mode for processing')
    parser.add_argument("--model", type=str, default="STAEformer", help="Model to train")
    parser.add_argument("--if_train", type=int, default=1, help="if train: 1 for train mode, 0 for test mode")
    args = parser.parse_args()

    # seed = torch.randint(1000, (1,)) # set random seed here
    seed_everything(2021)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"

    # -------------------------------- load model -------------------------------- #

    # model = STAEformer(**cfg["model_args"])

    if args.model in available_models:
        model_class = available_models[args.model]
        model_name = args.model
        if 'SRCA' in args.model:
            with open(f"SRCA.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            cfg = cfg[dataset]
            cfg["model_args"]["mode"] = args.mode
            model = model_class(**cfg["model_args"])
        else:
            with open(f"{args.model}.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            cfg = cfg[dataset]
            model = model_class(**cfg["model_args"])
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # 添加这行代码，将模型移动到指定设备
    model = model.to(DEVICE)
    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{dataset}")
    if not os.path.exists(log):
        os.makedirs(log)
    log = os.path.join(log, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # 测试模式逻辑
    if args.if_train == 0:
        # 测试模式 - 直接加载预训练模型
        model = model.to(DEVICE)

        # 查找已保存的模型文件
        model_files = []
        if os.path.exists(save_path):
            for file in os.listdir(save_path):
                if file.startswith(f"{model_name}-{dataset}") and file.endswith(".pt"):
                    model_files.append(file)

        if model_files:
            # 选择最新的模型文件
            model_files.sort(reverse=True)
            saved_model_path = os.path.join(save_path, model_files[0])
            print_log(f"Loading pre-trained model from: {saved_model_path}", log=log)
            model.load_state_dict(torch.load(saved_model_path, map_location=DEVICE))
        else:
            raise FileNotFoundError(f"No pre-trained model found for {model_name}-{dataset}")

        # 直接进行测试
        print_log("--------- Test Mode ---------", log=log)
        test_model(model, testset_loader, log=log)
        log.close()
        exit(0)
    else:
        # ---------------------- set loss, optimizer, scheduler ---------------------- #

        if 'SRCA' in args.model and cfg["loss"] == 'MAE':
            criterion = MaskedMAELoss()
        elif 'SRCA' in args.model and cfg["loss"] == 'HUBER':
            criterion = nn.HuberLoss()
            # print(criterion.delta)
            criterion.delta = cfg['delta']
        else:
            criterion = MaskedMAELoss()


        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0),
            eps=cfg.get("eps", 1e-8),
        )

        if 'SRCA' in args.model and cfg.get("optimizer", 'None') == 'multistep':
            print_log("MultiStepLR", log=log)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=cfg["milestones"],
                gamma=cfg.get("lr_decay_rate", 0.1),
                verbose=False,
            )
        elif  'SRCA' in args.model and cfg.get("optimizer", 'None') != 'multistep':
            print_log("ReduceLROnPlateau", log=log)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min', factor=cfg.get("lr_decay_rate", 0.1),
                patience=cfg.get("early_stop", 10),
                threshold=0.0001, threshold_mode='abs',
                cooldown=0, min_lr=2e-6, eps=1e-08)
        else:
            print_log("MultiStepLR", log=log)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=cfg["milestones"],
                gamma=cfg.get("lr_decay_rate", 0.1),
                verbose=False,
            )

        # --------------------------- print model structure -------------------------- #

        print_log("---------", model_name, "---------", log=log)
        print_log(
            json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
        )
        # 创建一个样本输入用于计算FLOPs
        sample_shape = [
            cfg["batch_size"],
            cfg["in_steps"],
            cfg["num_nodes"],
            next(iter(trainset_loader))[0].shape[-1],
        ]
        sample_input = torch.zeros(sample_shape).to(DEVICE)
        # 计算FLOPs和参数数量
        flops, params = calculate_flops(model, sample_input)
        print_log(f"FLOPs: {flops / 1e9:.2f}G", log=log)  # 转换为GFLOPs
        print_log(f"Params: {params / 1e6:.2f}M", log=log)  # 转换为百万参数

        print_log(
            summary(
                model,
                sample_shape,
                verbose=0,  # avoid print twice
            ),
            log=log,
        )
        print_log(log=log)

        # --------------------------- train and test model --------------------------- #

        print_log(f"Loss: {criterion._get_name()}", log=log)
        print_log(log=log)

        model = train(
            model,
            trainset_loader,
            valset_loader,
            testset_loader,
            optimizer,
            scheduler,
            criterion,
            clip_grad=cfg.get("clip_grad"),
            max_epochs=cfg.get("max_epochs", 200),
            early_stop=cfg.get("early_stop", 10),
            verbose=1,
            log=log,
            save=save,
        )

        print_log(f"Saved Model: {save}", log=log)

        test_model(model, testset_loader, log=log)

        log.close()
