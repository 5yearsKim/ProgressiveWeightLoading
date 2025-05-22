from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar

import torch
from datasets import load_dataset
from torch import nn
from torchvision import transforms as T
from transformers import PretrainedConfig, PreTrainedModel

from .prepare_data import CIFAR100TorchDataset


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
    model_for_image_classification: PreTrainedModel,
    model_config: PretrainedConfig,
) -> PreTrainedModel | None:
    if model_from is None:
        return None

    try:
        return model_for_image_classification.from_pretrained(model_from)
    except:
        try:
            config = model_config.from_pretrained(Path(model_from) / "config.json")
            return model_for_image_classification(config)
        except:
            raise ValueError(f"Invalid model_from value: {model_from}.")


def prepare_experiment(
    model_type: Literal["resnet", "lenet5"],
    teacher_from: str | PretrainedConfig | None = None,
    student_from: str | PretrainedConfig | None = None,
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

            train_transform = T.Compose(
                [
                    T.RandomCrop(32, padding=4, padding_mode="reflect"),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),  # [0,1]
                    T.Lambda(lambda t: t - 0.5),  # → roughly [–0.5, +0.5]
                ]
            )
            eval_transform = T.Compose(
                [
                    T.ToTensor(),  # [0,1]
                    T.Lambda(lambda t: t - 0.5),  # → roughly [–0.5, +0.5]
                ]
            )

            def train_preprocess(batch):
                return _preprocess(batch, train_transform)

            def eval_preprocess(batch):
                return _preprocess(batch, eval_transform)

            def _preprocess(batch, transform):
                tensor_list = []
                for arr in batch["img"]:
                    img = arr.convert("RGB")
                    tensor_list.append(transform(img))
                batch["pixel_values"] = torch.stack(tensor_list)
                batch["labels"] = batch["label"]
                return batch

            ds = load_dataset(DATASET_NAME)
            ds_train = ds["train"].map(
                train_preprocess,
                batched=True,
                batch_size=PREPROCESS_BATCH_SIZE,
                remove_columns=["img", "label"],
            )
            ds_val = ds["test"].map(
                eval_preprocess,
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

            ds_train = CIFAR100TorchDataset("train")
            ds_eval = CIFAR100TorchDataset("eval")

            e_set.dataset = ExperimentDataset(
                name=DATASET_NAME,
                train=ds_train,
                eval=ds_eval,
            )

    else:
        raise ValueError(f"{model_type} not defined")

    return e_set
