from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from pwl_model.core import SwapNet

from .custom_data import CIFAR10TorchDataset, CIFAR100TorchDataset
from .utils import load_block_model


@dataclass
class ExperimentDataset:
    name: str
    train: torch.utils.data.Dataset | None = None
    eval: torch.utils.data.Dataset | None = None
    collate_fn: Optional[Callable] | None = None


@dataclass
class ExperimentSet:
    teacher: nn.Module | None = None
    student: nn.Module | None = None
    swapnet: nn.Module | None = None


class ExperimentComposer:
    def prepare_model(
        self,
        model_type: Literal["resnet", "lenet5"],
        teacher_from: str | PretrainedConfig | None = None,
        student_from: str | PretrainedConfig | None = None,
        use_swapnet: bool = True,
    ):
        e_set = ExperimentSet()

        if model_type == "lenet5":

            from pwl_model.models.lenet5 import (
                BlockLeNet5Config, BlockLeNet5ForImageClassification)

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

                assert teacher is not None, "Teacher model is not loaded."
                assert student is not None, "Student model is not loaded."
                swapnet = SwapNet(
                    teacher=teacher,
                    student=student,
                    input_shape=INPUT_SHAPE,
                )
                e_set.swapnet = swapnet

        elif model_type == "resnet":

            from pwl_model.models.resnet import (
                BlockResNetConfig, BlockResNetForImageClassification)

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

        elif model_type == "vgg":

            from pwl_model.models.vgg import (BlockVGGConfig,
                                              BlockVGGForImageClassification)

            # Create the teacher and student models
            teacher = load_block_model(
                teacher_from, BlockVGGForImageClassification, BlockVGGConfig
            )
            student = load_block_model(
                student_from, BlockVGGForImageClassification, BlockVGGConfig
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

        else:
            raise ValueError(f"{model_type} not defined")

        return e_set

    def prepare_data(
        self, dataset_name: str, use_train: bool = True, use_eval: bool = True
    ) -> ExperimentDataset:
        """
        Prepare the data for the PWL model.
        """

        d_set = ExperimentDataset(name=dataset_name)

        if dataset_name == "cifar100":
            if use_train:
                d_set.train = CIFAR100TorchDataset(stage="train")
            if use_eval:
                d_set.eval = CIFAR100TorchDataset(stage="eval")

        elif dataset_name == "cifar10":
            if use_train:
                d_set.train = CIFAR10TorchDataset(stage="train")
            if use_eval:
                d_set.eval = CIFAR10TorchDataset(stage="eval")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        return d_set
