import argparse
import os

import mlflow
import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import (AutoImageProcessor, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)
from transformers.integrations import MLflowCallback
from transformers.trainer_utils import EvalPrediction

from pwl_model.feature_distiller import DistillerOutput, FeatureDistiller
from pwl_model.utils.training_utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a teacher into a smaller student."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "lenet"],
        default="lenet",
        help="Model type",
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        default="./ckpts/lenet-cifar10/teachers/checkpoint-7820",
        help="Path or model identifier of the pretrained teacher",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["zh-plus/tiny-imagenet", "uoft-cs/cifar10"],
        default="uoft-cs/cifar10",
        help="🤗 dataset identifier (e.g. 'zh-plus/tiny-imagenet')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ckpts/lenet-cifar10/student_training",
        help="Where to save distilled checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per device",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=bool,
        default=True,
        help="Batch size per device",
    )
    parser.add_argument(
        "--is_sample",
        action="store_true",
        help="Take only sample dataset for the development purpose",
    )
    return parser.parse_args()


meter_dict: dict[str, AverageMeter] = {
    "hard": AverageMeter(),
    "soft": AverageMeter(),
    "feat_sync": AverageMeter(),
    "feat_recon": AverageMeter(),
    "cross": AverageMeter(),
}


class DistilTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # standard HuggingFace Trainer calls model(...) under the hood
        outputs: DistillerOutput = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
        )
        loss = outputs["loss"]

        for key in meter_dict.keys():
            meter_dict[key].update(outputs[f"loss_{key}"].item())

        return (loss, outputs) if return_outputs else loss


class MeterCallback(TrainerCallback):
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # logs is the dict of already‐computed metrics (eval_loss, epoch, etc)
        logs = {}

        # Pull out your meters and add their epoch‐averages to the logs
        for name, meter in meter_dict.items():
            logs[f"{name}"] = format(meter.avg, ".3g")
            meter.reset()

        print(logs)

        # Tell HF to write these logs now
        control.should_log = True
        return control


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "resnet":
        from transformers import ResNetConfig, ResNetForImageClassification

        # 1. Load teacher & build student config
        teacher = ResNetForImageClassification.from_pretrained(args.teacher_path)
        teacher_cfg = teacher.config

        student_cfg = ResNetConfig(**teacher_cfg.to_dict())
        student_cfg.depths = [max(1, d // 2) for d in student_cfg.depths]
        student = ResNetForImageClassification(student_cfg)

        processor = AutoImageProcessor.from_pretrained(args.teacher_path)

        def preprocess(batch):
            inputs = processor(batch["image"], return_tensors="pt")
            batch["pixel_values"] = inputs["pixel_values"]
            return batch

    elif args.model_type == "lenet":
        from torchvision import transforms as T

        # from pwl_model.layers.block_net import BlockModelForImageClassification
        # from pwl_model.layers.swap_net import SwapNet
        # from pwl_model.lenet5.lenet5 import LeNet5Config, create_lenet5_blocks
        from pwl_model.models.lenet5 import (
            BlockLeNet5Config,
            BlockLeNet5ForImageClassification,
        )
        from pwl_model.core.swap_net import SwapNet

        mlflow.set_experiment("lenet5-cifar10-distill")
        mlflow.log_param("device", str(device))

        # t_config = BlockLeNet5Config()
        # teacher = BlockLeNet5ForImageClassification(t_config)

        # state_dict = load_file(os.path.join(args.teacher_path, "model.safetensors"))
        # teacher.load_state_dict(state_dict)
        teacher = BlockLeNet5ForImageClassification.from_pretrained(args.teacher_path)

        s_config = BlockLeNet5Config(cnn_channels=[3, 8], fc_sizes=[200, 120, 84])
        student = BlockLeNet5ForImageClassification(s_config)

        swapnet = SwapNet(
            teacher=teacher,
            student=student,
            input_shape=(3, 32, 32),
        )

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

    else:
        raise ValueError(f"{args.model_type} not defined")

    ds = load_dataset(args.dataset_name)
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
        remove_columns=[
            "img",
            "label",
        ],  # 'pixel_values' and 'labels' goes to input
    )
    if args.is_sample:
        ds_train = ds_train.select(range(500))
        ds_val = ds_val.select(range(100))

    distiller = FeatureDistiller(
        swapnet=swapnet,
    )

    def compute_metrics(p: EvalPrediction):
        logits = (
            p.predictions[0]
            if isinstance(p.predictions, (tuple, list))
            else p.predictions
        )
        preds = np.argmax(logits, axis=1)
        accuracy = (preds == p.label_ids).mean()
        return {"accuracy": accuracy}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        save_safetensors=True,
        report_to=["mlflow"],
        logging_dir="./runs",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,
    )

    trainer = DistilTrainer(
        model=distiller,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback(), MeterCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    main()
