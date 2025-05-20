import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from datasets import load_dataset
from torch import nn


@dataclass
class ExperimentDataset:
    name: str
    train: torch.utils.data.Dataset
    eval: torch.utils.data.Dataset


@dataclass
class ExperimentSet:
    teacher: nn.Module | None = None
    student: nn.Module | None = None
    swapnet: nn.Module | None = None

    dataset: ExperimentDataset | None = None


def looks_like_checkpoint_dir(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    # look for any .bin or .safetensors files in the top level
    has_bin = any(p.glob("*.bin"))
    has_safe = any(p.glob("*.safetensors"))
    return has_bin or has_safe


def prepare_experiment(
    model_type: Literal["resnet", "lenet5"],
    teacher_from: str | None = None,
    student_from: str | None = None,
    use_swapnet: bool = True,
    use_dataset: bool = True,
):
    e_set = ExperimentSet()

    if model_type == "lenet5":
        from torchvision import transforms as T

        from pwl_model.core import SwapNet
        from pwl_model.models.lenet5 import (BlockLeNet5Config,
                                             BlockLeNet5ForImageClassification)

        def load_model(model_from: str | None):
            if model_from is None:
                return None
            elif not os.path.exists(model_from) or not os.path.isdir(model_from):
                raise ValueError(f"Invalid model_from value: {model_from}. ")
            elif looks_like_checkpoint_dir(model_from):
                return BlockLeNet5ForImageClassification.from_pretrained(model_from)
            elif os.path.exists(os.path.join(model_from, "config.json")):
                config = BlockLeNet5Config.from_pretrained(model_from)
                return BlockLeNet5ForImageClassification(config)
            else:
                raise ValueError(f"Invalid model_from value: {model_from}.")

        # Create the teacher and student models
        teacher = load_model(teacher_from)
        student = load_model(student_from)
        e_set.teacher = teacher
        e_set.student = student

        if use_swapnet:
            INPUT_SHAPE = (3, 32, 32)
            swapnet = SwapNet(
                teacher=teacher,
                student=student,
                input_shape=INPUT_SHAPE,
            )
            e_set.swapnet = swapnet

        # create dataset
        if use_dataset:
            DATASET_NAME = "uoft-cs/cifar10"

            transform = T.Compose(
                [
                    T.ToTensor(),  # [0,1]
                    T.Lambda(lambda t: t - 0.5),  # → roughly [–0.5, +0.5]
                ]
            )

            def preprocess(batch):
                tensor_list = []
                for arr in batch["img"]:
                    img = arr.convert("RGB")
                    tensor_list.append(transform(img))
                batch["pixel_values"] = torch.stack(tensor_list)
                batch["labels"] = batch["label"]
                return batch

            ds = load_dataset(DATASET_NAME)
            ds_train = ds["train"].map(
                preprocess,
                batched=True,
                batch_size=32,
                remove_columns=["img", "label"],
            )
            ds_val = ds["test"].map(
                preprocess,
                batched=True,
                batch_size=32,
                # 'pixel_values' and 'labels' goes to input # 'pixel_values' and 'labels' goes to input
                remove_columns=[
                    "img",
                    "label",
                ],
            )
            e_set.dataset = ExperimentDataset(
                name=DATASET_NAME,
                train=ds_train,
                eval=ds_val,
            )
    elif model_type == "resnet":
        raise NotImplementedError(
            "ResNet model is not implemented yet. Please use lenet5 model."
        )
    else:
        raise ValueError(f"{model_type} not defined")

    return e_set
