import argparse
from typing import Literal

import mlflow
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, Trainer, TrainingArguments
from transformers.integrations import MLflowCallback


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
        "--model_path",
        type=str,
        # default="./ckpts/lenet-cifar10/teachers/checkpoint-7429",
        default="./ckpts/lenet-cifar10/students/converted_student",
        help="Path or model identifier of the model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["zh-plus/tiny-imagenet", "uoft-cs/cifar10"],
        default="uoft-cs/cifar10",
        help="🤗 dataset identifier (e.g. 'zh-plus/tiny-imagenet')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size per device",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "resnet":
        from transformers import ResNetConfig, ResNetForImageClassification

        model = ResNetForImageClassification.from_pretrained(args.teacher_path)

        processor = AutoImageProcessor.from_pretrained(args.teacher_path)

        def preprocess(batch):
            inputs = processor(batch["image"], return_tensors="pt")
            batch["pixel_values"] = inputs["pixel_values"]
            return batch

    elif args.model_type == "lenet":
        from torchvision import transforms as T

        from pwl_model.lenet5 import LeNet5ForImageClassification

        model = LeNet5ForImageClassification.from_pretrained(args.model_path).to(device)

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
    ds_val = ds["test"].map(
        preprocess,
        batched=True,
        batch_size=32,
        remove_columns=[
            "img",
            "label",
        ],  # 'pixel_values' and 'labels' goes to input
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).astype(np.float32).mean().item()}

    eval_args = TrainingArguments(
        output_dir="./tmp_eval",  # not used beyond logging
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        logging_strategy="no",  # you can silence logs if you like
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print(f"Evaluation results: {metrics}")


if __name__ == "__main__":
    main()
