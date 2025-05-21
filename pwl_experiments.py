import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar

import torch
from datasets import load_dataset
from torch import nn
from torchvision import datasets
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


def load_model(
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
        from torchvision import transforms as T

        from pwl_model.core import SwapNet
        from pwl_model.models.lenet5 import (BlockLeNet5Config,
                                             BlockLeNet5ForImageClassification)

        # Create the teacher and student models
        teacher = load_model(
            teacher_from, BlockLeNet5ForImageClassification, BlockLeNet5Config
        )
        student = load_model(
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
        teacher = load_model(
            teacher_from, BlockResNetForImageClassification, BlockResNetConfig
        )
        student = load_model(
            student_from, BlockResNetForImageClassification, BlockResNetConfig
        )
        e_set.teacher = teacher
        e_set.student = student

        if use_swapnet:
            INPUT_SHAPE = (3, 224, 224)
            assert teacher is not None, "Teacher model is not loaded."
            assert student is not None, "Student model is not loaded."
            swapnet = SwapNet(
                teacher=teacher,
                student=student,
                input_shape=INPUT_SHAPE,
            )
            e_set.swapnet = swapnet

        if use_dataset:
            image_processor = AutoImageProcessor.from_pretrained(
                "microsoft/resnet-18", use_fast=True
            )

            def hf_transform(pil_img):
                # image_processor returns pixel_values shape=(1, C, H, W)
                pv = image_processor(
                    images=[pil_img.convert("RGB")], return_tensors="pt"
                )["pixel_values"]
                return pv.squeeze(0)  # → Tensor[C,H,W]

            ds_train = datasets.ImageFolder(
                "data/imagenet/train",
                transform=hf_transform,
            )
            ds_val = datasets.ImageFolder("data/imagenet/val", transform=hf_transform)

            def collate_fn(batch):
                pixel_vals, labels = zip(*batch)

                # stack into one tensor
                pixel_values = torch.stack(pixel_vals, dim=0)  # → [B, C, H, W]
                labels = torch.tensor(labels, dtype=torch.long)  # → [B]

                return {"pixel_values": pixel_values, "labels": labels}

            # PREPROCESS_BATCH_SIZE = 128
            # ds = load_dataset("imagefolder", data_dir="./data/imagenet", )

            # def preprocess(batch):
            #     # examples["image"] is a list of PIL images
            #     outputs = image_processor(
            #         images=[img.convert("RGB") for img in batch["image"]],
            #         return_tensors="pt",
            #     )
            #     return {"pixel_values": outputs["pixel_values"], "labels": batch["label"]}

            # ds_train = ds["train"].map(
            #     preprocess,
            #     batched=True,
            #     batch_size=PREPROCESS_BATCH_SIZE,
            #     remove_columns=["image", "label"],
            # )

            # ds_val = ds["validation"].map(
            #     preprocess,
            #     batched=True,
            #     batch_size=PREPROCESS_BATCH_SIZE,
            #     remove_columns=["image", "label"],
            # )

            e_set.dataset = ExperimentDataset(
                name="imagenet",
                train=ds_train,
                eval=ds_val,
                collate_fn=collate_fn,
            )

    else:
        raise ValueError(f"{model_type} not defined")

    return e_set
