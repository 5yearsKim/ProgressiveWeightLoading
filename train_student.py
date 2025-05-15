import argparse

import mlflow
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, Trainer, TrainingArguments
from transformers.integrations import MLflowCallback

from pwl_model.feature_distiller import FeatureDistiller


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
        default="./ckpts/lenet-cifar10/archives/teacher-14858",
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
        default="./ckpts/lenet-cifar10/students",
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
    return parser.parse_args()


class DistilTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # standard HuggingFace Trainer calls model(...) under the hood
        loss_dict = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
        )
        loss = loss_dict["loss"]
        return (loss, loss_dict) if return_outputs else loss


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

        from pwl_model.lenet5 import LeNet5Config, LeNet5ForImageClassification

        mlflow.set_experiment("lenet5-cifar10-distill")
        mlflow.log_param("device", str(device))

        teacher = LeNet5ForImageClassification.from_pretrained(
            "./ckpts/lenet-cifar10/archives/checkpoint-7429"
        ).to(device)

        config = LeNet5Config()
        student = LeNet5ForImageClassification(config).to(device)

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

    distiller = FeatureDistiller(student, teacher)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": (preds == p.label_ids).mean()}

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
        greater_is_better=True,
        save_total_limit=1,
    )

    trainer = DistilTrainer(
        model=distiller,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    main()
