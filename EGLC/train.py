import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders, concatdataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
import warnings

warnings.filterwarnings("ignore")

def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, pyramid = model(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        # loss2 = Dice_loss(output1, target) + BCE_loss(torch.sigmoid(output1), target)
        # loss = 0.8*loss1 + 0.2*loss2
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output, pyramid = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(perf_accumulator), np.std(perf_accumulator)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        img_path = args.root + "/images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "/masks/*"
        target_paths = sorted(glob.glob(depth_path))

        train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
            input_paths, target_paths, batch_size=args.batch_size, imgsize=args.img_size
        )

    elif args.dataset == "CVC":
        img_path = args.root + "/Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "/Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))

        train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
            input_paths, target_paths, batch_size=args.batch_size, imgsize=args.img_size
        )

    elif args.dataset == "ISIC":
        img_path = args.root + "/images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "/masks/*"
        target_paths = sorted(glob.glob(depth_path))

        train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
            input_paths, target_paths, batch_size=args.batch_size, imgsize=args.img_size
        )

    elif args.dataset == "DSB":
        img_path = args.root + "/images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "/masks/*"
        target_paths = sorted(glob.glob(depth_path))

        train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
            input_paths, target_paths, batch_size=args.batch_size, imgsize=args.img_size
        )

    elif args.dataset == "Polyp":
        img_path1 = 'D:/dataset/Kvasir-SEG' + "/images/*"
        input_paths1 = sorted(glob.glob(img_path1))
        depth_path1 = 'D:/dataset/Kvasir-SEG' + "/masks/*"
        target_paths1 = sorted(glob.glob(depth_path1))

        img_path2 = 'D:/dataset/CVC-ClinicDB' + "/Original/*"
        input_paths2 = sorted(glob.glob(img_path2))
        depth_path2 = 'D:/dataset/CVC-ClinicDB' + "/Ground Truth/*"
        target_paths2 = sorted(glob.glob(depth_path2))

        train_dataloader, val_dataloader = concatdataloaders.get_dataloaders(
            input_paths1, input_paths2, target_paths1, target_paths2, batch_size=args.batch_size
        )



    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()

    perf = performance_metrics.DiceScore()

    model = models.EGIC(size=args.img_size)


    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    return (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    )


def train(args):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    ) = build(args)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss
            )
            test_measure_mean, test_measure_std = test(
                model, device, val_dataloader, epoch, perf
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                "Trained models/EGIC_" + "{}_".format(args.img_size) + args.dataset + ".pt",
            )
            prev_best_test = test_measure_mean



def get_args():
    parser = argparse.ArgumentParser(description="Train EGIC on specified dataset")
    parser.add_argument("--dataset", default="Kvasir", type=str, choices=["Kvasir", "CVC", "ISIC", "DSB", "Polyp"])
    parser.add_argument("--data-root", default="D:/dataset/Kvasir-SEG", type=str, dest="root")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )

    return parser.parse_args()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
