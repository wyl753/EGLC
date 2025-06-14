import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn

from Data import dataloaders, dataloaders_kvasir, dataloaders1
from Models import models
from Metrics import performance_metrics


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.type == "1":

        if args.test_dataset == "ISIC":
            img_path = args.root + "/images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "/masks/*"
            target_paths = sorted(glob.glob(depth_path))

        elif args.test_dataset == "DSB":
            img_path = args.root + "/images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "/masks/*"
            target_paths = sorted(glob.glob(depth_path))

        elif args.test_dataset == "Kvasir":
            img_path = args.root + "/images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "/masks/*"
            target_paths = sorted(glob.glob(depth_path))

        _, test_dataloader, _ = dataloaders.get_dataloaders(
            input_paths, target_paths, batch_size=1, imgsize=args.img_size
        )

        _, test_indices, _ = dataloaders.split_ids(len(target_paths))
        target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

    elif args.type == "2":
        if args.test_dataset == "Kvasir":
            img_path = args.root + "/images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "/masks/*"
            target_paths = sorted(glob.glob(depth_path))

            _, test_dataloader = dataloaders_kvasir.get_dataloaders(
                input_paths, target_paths, batch_size=1
            )
            _, test_indices = dataloaders_kvasir.split_ids(len(target_paths))
            target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]
        elif args.test_dataset == "ClinicDB":
            img_path = args.root + "/Original/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "/Ground Truth/*"
            target_paths = sorted(glob.glob(depth_path))

            _, test_dataloader = dataloaders_kvasir.get_dataloaders(
                input_paths, target_paths, batch_size=1
            )
            _, test_indices = dataloaders_kvasir.split_ids(len(target_paths))
            target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

        elif args.test_dataset == "ColonDB":
            img_path = args.root + "/images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "/masks/*"
            target_paths = sorted(glob.glob(depth_path))

            test_dataloader = dataloaders1.get_test_dataloaders(
                input_paths, target_paths, batch_size=1
            )
            test_indices = np.linspace(0, len(input_paths) - 1, len(input_paths)).astype("int")
            target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

        elif args.test_dataset == "ETIS":
            img_path = args.root + "/images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "/masks/*"
            target_paths = sorted(glob.glob(depth_path))

            test_dataloader = dataloaders1.get_test_dataloaders(
                input_paths, target_paths, batch_size=1
            )
            test_indices = np.linspace(0, len(input_paths) - 1, len(input_paths)).astype("int")
            target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

        elif args.test_dataset == "Endo":
            img_path = args.root + "/images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "/masks/*"
            target_paths = sorted(glob.glob(depth_path))

            test_dataloader = dataloaders1.get_test_dataloaders(
                input_paths, target_paths, batch_size=1
            )
            test_indices = np.linspace(0, len(input_paths) - 1, len(input_paths)).astype("int")
            target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

    perf = performance_metrics.DiceScore()

    model = models.EGIC(size=args.img_size)

    state_dict = torch.load(
        "./Trained models/EGIC_{}_{}.pt".format(args.img_size, args.train_dataset)
    )
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, test_dataloader, perf, model, target_paths


@torch.no_grad()
def predict(args):
    device, test_dataloader, perf_measure, model, target_paths = build(args)

    if not os.path.exists("./Predictions"):
        os.makedirs("./Predictions")
    if not os.path.exists("./Predictions/Trained on {}".format(args.train_dataset)):
        os.makedirs("./Predictions/Trained on {}".format(args.train_dataset))
    if not os.path.exists(
        "./Predictions/Trained on {}/Tested on {}".format(
            args.train_dataset, args.test_dataset
        )
    ):
        os.makedirs(
            "./Predictions/Trained on {}/Tested on {}".format(
                args.train_dataset, args.test_dataset
            )
        )
    if not os.path.exists("./gt"):
        os.makedirs("./gt")
    if not os.path.exists("./gt/{}".format(args.test_dataset)):
        os.makedirs("./gt/{}".format(args.test_dataset))

    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        output, x1 = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        target_paths[i] = target_paths[i].rsplit('.', 1)[0] + '.jpg'
        gt_map = np.array(target.cpu())
        gt_map = np.squeeze(gt_map)
        gt_map = gt_map > 0
        cv2.imwrite(
            "./Predictions/Trained on {}/Tested on {}/{}".format(
                args.train_dataset, args.test_dataset, os.path.basename(target_paths[i])
            ),
            predicted_map * 255,
        )
        cv2.imwrite(
            "./gt/{}/{}".format(
                args.test_dataset, os.path.basename(target_paths[i])
            ),
            gt_map * 255,
        )
        if i + 1 < len(test_dataloader):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument("--type", default="1", type=str, choices=["1", "2", "3"])
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument(
        "--train-dataset", default="kvasir", type=str
    )
    parser.add_argument(
        "--test-dataset", default="Kvasir", type=str
    )
    parser.add_argument("--data-root", default='D:/dataset/Kvasir-SEG', type=str,  dest="root")

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()

