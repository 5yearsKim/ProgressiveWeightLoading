import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar

import torch
from datasets import load_dataset
from torch import nn
from torchvision import transforms as T
from transformers import AutoImageProcessor

TemplateModel = TypeVar("TM")
TemplateConfig = TypeVar("TC")


@dataclass
class ExperimentDataset:
    name: str
    train: torch.utils.data.Dataset
    eval: torch.utils.data.Dataset
    collate_fn: Optional[Callable] | None = None


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


def load_block_model(
    model_from: str | None,
    model_for_image_classification: TemplateModel,
    model_config: TemplateConfig,
) -> TemplateModel | None:
    if model_from is None:
        return None
    elif not os.path.exists(model_from) or not os.path.isdir(model_from):
        raise ValueError(f"Invalid model_from value: {model_from}. ")
    elif looks_like_checkpoint_dir(model_from):
        return model_for_image_classification.from_pretrained(model_from)
    elif os.path.exists(os.path.join(model_from, "config.json")):
        config = model_config.from_pretrained(model_from)
        return model_for_image_classification(config)
    else:
        raise ValueError(f"Invalid model_from value: {model_from}.")


def prepare_experiment(
    model_type: Literal["resnet", "lenet5"],
    teacher_from: str | None = None,
    student_from: str | None = None,
    use_swapnet: bool = True,
    use_dataset: bool = True,
):
    e_set = ExperimentSet()

    if model_type == "lenet5":

        from pwl_model.core import SwapNet
        from pwl_model.models.lenet5 import (BlockLeNet5Config,
                                             BlockLeNet5ForImageClassification)

        # Create the teacher and student models
        teacher = load_block_model(
            teacher_from, BlockLeNet5ForImageClassification, BlockLeNet5Config
        )
        student = load_block_model(
            student_from, BlockLeNet5ForImageClassification, BlockLeNet5Config
        )
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
            PREPROCESS_BATCH_SIZE = 32

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
                batch_size=PREPROCESS_BATCH_SIZE,
                remove_columns=["img", "label"],
            )
            ds_val = ds["test"].map(
                preprocess,
                batched=True,
                batch_size=PREPROCESS_BATCH_SIZE,
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
        from torchvision import transforms as T

        from pwl_model.core import SwapNet
        from pwl_model.models.resnet import (BlockResNetConfig,
                                             BlockResNetForImageClassification)

        # Create the teacher and student models
        teacher = load_block_model(
            teacher_from, BlockResNetForImageClassification, BlockResNetConfig
        )
        student = load_block_model(
            student_from, BlockResNetForImageClassification, BlockResNetConfig
        )
        e_set.teacher = teacher
        e_set.student = student

        if use_swapnet:
            INPUT_SHAPE = (3, 32, 32)
            assert teacher is not None, "Teacher model is not loaded."
            assert student is not None, "Student model is not loaded."
            swapnet = SwapNet(
                teacher=teacher,
                student=student,
                input_shape=INPUT_SHAPE,
            )
            e_set.swapnet = swapnet

        if use_dataset:
            DATASET_NAME = "cifar100"
            PREPROCESS_BATCH_SIZE = 64

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
                batch["labels"] = batch["fine_label"]
                return batch

            ds = load_dataset(DATASET_NAME)

            ds_train = ds["train"].map(
                preprocess,
                batched=True,
                batch_size=PREPROCESS_BATCH_SIZE,
                remove_columns=["img", "fine_label", "coarse_label"],
            )
            ds_val = ds["test"].map(
                preprocess,
                batched=True,
                batch_size=PREPROCESS_BATCH_SIZE,
                remove_columns=["img", "fine_label", "coarse_label"],
            )
            e_set.dataset = ExperimentDataset(
                name=DATASET_NAME,
                train=ds_train,
                eval=ds_val,
            )

            e_set.dataset = ExperimentDataset(
                name=DATASET_NAME,
                train=ds_train,
                eval=ds_val,
            )

    else:
        raise ValueError(f"{model_type} not defined")

    return e_set
