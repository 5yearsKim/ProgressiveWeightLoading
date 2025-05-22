from typing import Literal

from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms as T


class CIFAR100TorchDataset(Dataset):
    MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    def __init__(self, stage: Literal["train", "eval"]):
        self.stage = stage
        if stage == "train":
            self.ds = load_dataset("cifar100", split="train")
            self.transform = T.Compose(
                [
                    T.RandomRotation(15),
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, padding=4, padding_mode="reflect"),
                    T.ToTensor(),  # [0,1]
                    T.Normalize(self.MEAN, self.STD),
                ]
            )

        elif stage == "eval":
            self.ds = load_dataset("cifar100", split="test")
            self.transform = T.Compose(
                [T.ToTensor(), T.Normalize(self.MEAN, self.STD)]  # [0,1]
            )
        else:
            raise ValueError(f"stage {stage} not supported")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["img"].convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "labels": item["fine_label"],
        }


class CIFAR10TorchDataset(Dataset):
    # CIFAR-10 normalization stats
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)

    def __init__(self, stage: Literal["train", "eval"]):
        if stage == "train":
            self.ds = load_dataset("cifar10", split="train")
            self.transform = T.Compose(
                [
                    T.RandomRotation(15),
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, padding=4, padding_mode="reflect"),
                    T.ToTensor(),
                    T.Normalize(self.MEAN, self.STD),
                ]
            )
        elif stage == "eval":
            self.ds = load_dataset("cifar10", split="test")
            self.transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(self.MEAN, self.STD),
                ]
            )
        else:
            raise ValueError(f"stage {stage!r} not supported")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["img"].convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "labels": item["label"],
        }
