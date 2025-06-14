import numpy as np
import random
import multiprocessing

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data
from torch.utils.data import ConcatDataset
from Data.dataset import SegDataset



def split_ids(len_ids):
    train_size = int(round((90 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    return train_indices, test_indices

def get_dataloaders(input_paths,input_paths1, target_paths,target_paths1, batch_size):

    transform_input4train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((256, 256)), transforms.Grayscale()]
    )

    train_dataset1 = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4train,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=True,
    )
    train_dataset2 = SegDataset(
        input_paths=input_paths1,
        target_paths=target_paths1,
        transform_input=transform_input4train,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=True,
    )

    val_dataset1 = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    val_dataset2 = SegDataset(
        input_paths=input_paths1,
        target_paths=target_paths1,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    train_indices, test_indices = split_ids(len(input_paths))
    train_indices1, test_indices1 = split_ids(len(input_paths1))

    train_dataset1 = data.Subset(train_dataset1, train_indices)
    train_dataset2 = data.Subset(train_dataset2, train_indices1)


    val_dataset1 = data.Subset(val_dataset1, test_indices)
    val_dataset2 = data.Subset(val_dataset2, test_indices1)

    train_dataset = ConcatDataset([train_dataset1, train_dataset2])
    val_dataset = ConcatDataset([val_dataset1, val_dataset2])

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    return train_dataloader, val_dataloader



