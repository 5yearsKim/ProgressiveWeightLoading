import random
from abc import ABC, abstractmethod
from typing import Literal

from datasets import Image as HFDatasetsImage
from datasets import load_dataset
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision import transforms as T


class CIFARTorchDataset(ABC, Dataset):
    # CIFAR-10 normalization stats
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)
    LABEL_KEY = "label"
    IMAMGE_KEY = "img"

    def __init__(self, stage: Literal["train", "eval"], reshape_size=None):
        self.size = self._get_input_size(reshape_size)

        transforms = []

        if reshape_size is not None:
            transforms.append(T.Resize(self.size))

        if stage == "train":
            self.ds = self.get_dataset("train")

            pad = (
                self.size // 8
                if isinstance(self.size, int)
                else tuple(s // 8 for s in self.size)
            )

            transforms.extend(
                [
                    T.RandomRotation(15),
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(self.size, padding=pad, padding_mode="reflect"),
                ]
            )
        elif stage == "eval":
            self.ds = self.get_dataset("test")
        else:
            raise ValueError(f"stage {stage!r} not supported")

        transforms.extend(
            [
                T.ToTensor(),
                T.Normalize(self.MEAN, self.STD),
            ]
        )

        self.transform = T.Compose(transforms)

    @abstractmethod
    def get_dataset(self, split: str):
        if split == "train":
            raise NotImplementedError()
        elif split == "test":
            raise NotImplementedError()
        else:
            raise ValueError()

    def _get_input_size(self, reshape_size):
        if reshape_size is None:
            size = 32
        elif isinstance(reshape_size, int):
            size = reshape_size
        elif isinstance(reshape_size, tuple):
            if len(reshape_size) == 2:
                size = reshape_size
            elif len(reshape_size) == 3:
                # drop the channel dim
                _, h, w = reshape_size
                size = (h, w)
            else:
                raise ValueError(
                    f"reshape_size tuple must be len 2 or 3, got {reshape_size}"
                )
        return size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        img = item[self.IMAMGE_KEY].convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "labels": item[self.LABEL_KEY],
        }


class CIFAR10TorchDataset(CIFARTorchDataset):
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)
    LABEL_KEY = "label"
    IMAMGE_KEY = "img"

    def get_dataset(self, split: str):
        if split == "train":
            return load_dataset("cifar10", split="train")
        elif split == "test":
            return load_dataset("cifar10", split="test")
        else:
            raise ValueError()


class CIFAR100TorchDataset(CIFARTorchDataset):
    MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    LABEL_KEY = "fine_label"
    IMAMGE_KEY = "img"

    def get_dataset(self, split: str):
        if split == "train":
            return load_dataset("cifar100", split="train")
        elif split == "test":
            return load_dataset("cifar100", split="test")
        else:
            raise ValueError()


class ImageNetDataset(Dataset):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    LABEL_KEY = "label"
    IMAGE_KEY = "image"

    def __init__(self, stage: Literal["train", "eval"], reshape_size=None):
        assert stage in ["train", "eval"], f"Invalid stage {stage}"
        self.size = reshape_size or 224

        split = "train" if stage == "train" else "validation"
        ds = load_dataset("imagenet-1k", split=split, encoding="utf-16")

        self.ds = ds

        transforms = [
            T.Resize(256),
            T.CenterCrop(self.size),
        ]

        if stage == "train":
            pad = (
                self.size // 8
                if isinstance(self.size, int)
                else tuple(s // 8 for s in self.size)
            )
            transforms += [
                T.RandomRotation(15),
                T.RandomHorizontalFlip(),
                T.RandomCrop(self.size, padding=pad, padding_mode="reflect"),
            ]

        transforms += [
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ]
        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        path = item[self.IMAGE_KEY]["path"]
        img = PILImage.open(path).convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "labels": item[self.LABEL_KEY],
        }
