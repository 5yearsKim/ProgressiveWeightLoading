import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from datasets import Image as DatasetsImage
from datasets import load_dataset
from PIL import Image as PILImage
from torch import nn
from transformers import AutoImageProcessor, PretrainedConfig, PreTrainedModel

from pwl_model.core import SwapNet

from .custom_data import (CIFAR10TorchDataset, CIFAR100TorchDataset,
                          ImageNetDataset)
from .utils import looks_like_checkpoint_dir


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
    def __init__(
        self,
        model_type: str,
        dataset_name: Literal["cifar10", "cifar100", "imagenet"],
    ) -> None:
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.input_shape = self._get_input_shape()
        self.num_labels = 10 if dataset_name == "cifar10" else 100

    def _get_input_shape(self) -> tuple:
        if self.model_type == "vit":
            return (3, 224, 224)
        return (3, 32, 32)

    def _load_block_model(
        self,
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
                config = model_config.from_pretrained(
                    Path(model_from) / "config.json",
                )
                return model_for_image_classification(config)
            except Exception as e:
                print(e)
                raise ValueError(f"Invalid model_from value: {model_from}")

    def prepare_model(
        self,
        teacher_from: str | PretrainedConfig | None = None,
        student_from: str | PretrainedConfig | None = None,
        use_swapnet: bool = True,
    ):
        e_set = ExperimentSet()

        if self.model_type == "lenet5":

            from pwl_model.models.lenet5 import (
                BlockLeNet5Config, BlockLeNet5ForImageClassification)

            # Create the teacher and student models
            teacher = self._load_block_model(
                teacher_from,
                BlockLeNet5ForImageClassification,
                BlockLeNet5Config,
                num_labels=self.num_labels,
            )
            student = self._load_block_model(
                student_from,
                BlockLeNet5ForImageClassification,
                BlockLeNet5Config,
                num_labels=self.num_labels,
            )

        elif self.model_type == "resnet":

            from pwl_model.models.resnet import (
                BlockResNetConfig, BlockResNetForImageClassification)

            # Create the teacher and student models
            teacher = self._load_block_model(
                teacher_from,
                BlockResNetForImageClassification,
                BlockResNetConfig,
            )
            student = self._load_block_model(
                student_from, BlockResNetForImageClassification, BlockResNetConfig
            )

        elif self.model_type == "vgg":

            from pwl_model.models.vgg import (BlockVGGConfig,
                                              BlockVGGForImageClassification)

            # Create the teacher and student models
            teacher = self._load_block_model(
                teacher_from, BlockVGGForImageClassification, BlockVGGConfig
            )
            student = self._load_block_model(
                student_from, BlockVGGForImageClassification, BlockVGGConfig
            )
        elif self.model_type == "vit":
            from pwl_model.models.vit import (BlockViTConfig,
                                              BlockViTForImageClassification)

            # Create the teacher and student models
            teacher = self._load_block_model(
                teacher_from, BlockViTForImageClassification, BlockViTConfig
            )
            student = self._load_block_model(
                student_from, BlockViTForImageClassification, BlockViTConfig
            )

        else:
            raise ValueError(f"{self.model_type} not defined")

        e_set.teacher = teacher
        e_set.student = student

        if use_swapnet:

            assert teacher is not None, "Teacher model is not loaded."
            assert student is not None, "Student model is not loaded."
            swapnet = SwapNet(
                teacher=teacher,
                student=student,
                input_shape=self.input_shape,
                channel_last=self.model_type in ["vit"],
            )
            e_set.swapnet = swapnet

        return e_set

    def prepare_data(
        self, use_train: bool = True, use_eval: bool = True
    ) -> ExperimentDataset:
        """
        Prepare the data for the PWL model.
        """

        dataset_name = self.dataset_name

        d_set = ExperimentDataset(name=dataset_name)

        if dataset_name == "cifar100":
            reshape_size = None if self.input_shape == (3, 32, 32) else self.input_shape
            if use_train:
                d_set.train = CIFAR100TorchDataset(
                    stage="train", reshape_size=reshape_size
                )
            if use_eval:
                d_set.eval = CIFAR100TorchDataset(
                    stage="eval", reshape_size=reshape_size
                )

        elif dataset_name == "cifar10":
            reshape_size = None if self.input_shape == (3, 32, 32) else self.input_shape
            if use_train:
                d_set.train = CIFAR10TorchDataset(
                    stage="train", reshape_size=reshape_size
                )
            if use_eval:
                d_set.eval = CIFAR10TorchDataset(
                    stage="eval", reshape_size=reshape_size
                )
        elif dataset_name == "imagenet":

            ds = load_dataset("imagenet-1k")
            ds = ds.cast_column("image", DatasetsImage(decode=False))

            # 2. Prepare feature extractor / image processor
            processor = AutoImageProcessor.from_pretrained(
                # "WinKawaks/vit-tiny-patch16-224"
                "google/vit-base-patch16-224-in21k",
                use_fast=True,
            )

            def preprocess_fn(examples):

                def _decode_image(b, dummy_size=(224, 224), dummy_color=(0, 0, 0)):
                    # b is a dict with 'bytes' and/or 'path'
                    try:
                        img = PILImage.open(io.BytesIO(b["bytes"]))
                    except Exception as e1:
                        print("in-memory decode failed:", e1)
                        try:
                            img = PILImage.open(b["path"])
                        except Exception as e2:
                            print("disk decode failed:", e2)
                            # give them a dummy RGB image instead
                            img = PILImage.new("RGB", dummy_size, color=dummy_color)
                    # finally, ensure it’s RGB
                    return img.convert("RGB")

                imgs = []
                for b in examples["image"]:
                    imgs.append(_decode_image(b))
                out = processor(images=imgs, return_tensors="pt")
                return {
                    "pixel_values": out.pixel_values,  # shape (batch,3,H,W)
                    "labels": examples["label"],  # list of ints
                }

            train_ds = ds["train"].with_transform(preprocess_fn)
            eval_ds = ds["validation"].with_transform(preprocess_fn)

            if use_train:
                d_set.train = train_ds
            if use_eval:
                d_set.eval = eval_ds

        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        return d_set
