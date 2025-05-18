import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, Trainer, TrainingArguments

from pwl_model.feature_distiller import FeatureDistiller
from pwl_model.swap_net import SwapNet


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
        default="./ckpts/lenet-cifar10/students/checkpoint-17193",
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

        from pwl_model.layers.block_module import \
            BlockModelForImageClassification
        from pwl_model.lenet5 import LeNet5Config, create_lenet5_blocks

        t_blocks, last_out_dim = create_lenet5_blocks(LeNet5Config())
        teacher = BlockModelForImageClassification(
            blocks=t_blocks,
            last_out_dim=last_out_dim,
            num_labels=10,
        ).to(device)

        s_blocks, last_out_dim = create_lenet5_blocks(
            LeNet5Config(cnn_channels=[3, 8], fc_sizes=[200, 120, 84])
        )
        student = BlockModelForImageClassification(
            blocks=s_blocks,
            last_out_dim=last_out_dim,
            num_labels=10,
        ).to(device)

        swapnet = SwapNet(
            teacher=teacher,
            student=student,
            input_shape=(3, 32, 32),
        )
        distiller = FeatureDistiller(swapnet=swapnet)

        load_model(distiller, os.path.join(args.model_path, "model.safetensors"))

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

    def collate_fn(batch):
        pixels = torch.stack(
            [torch.as_tensor(example["pixel_values"]) for example in batch], dim=0
        )
        labels = torch.tensor(
            [example["labels"] for example in batch], dtype=torch.long
        )
        return {"pixel_values": pixels, "labels": labels}

    eval_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    swapnet.to(device)
    swapnet.eval()

    from_teachers = [True] * swapnet.num_blocks

    def get_accuracy(from_teachers: list[bool]):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                x = batch["pixel_values"].to(device)
                y = batch["labels"].to(device)

                logits = swapnet(x, from_teachers)

                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        # 6) Compute metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        accuracy = (all_preds == all_labels).mean()
        return accuracy

    num_blocks = swapnet.num_blocks

    candidates: list[list[bool]] = [
        [True] * num_blocks,
        [False] * num_blocks,
    ]

    for i in range(1, num_blocks):
        candidates.append([True] * i + [False] * (num_blocks - i))
    for i in range(1, num_blocks):
        candidates.append([False] * i + [True] * (num_blocks - i))

    for from_teachers in candidates:
        accuracy = get_accuracy(from_teachers)
        print(
            f"{[('t' if item else 's' ) for item in from_teachers]}, accuracy: {accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
