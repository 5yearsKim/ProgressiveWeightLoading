import argparse
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pwl_model.core import FeatureDistiller, SwapNet
from pwl_model.lab import ExperimentComposer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a teacher into a smaller student."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "lenet5", "vgg"],
        help="Model type",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["cifar10", "cifar100"],
        help="Data Type",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path or model identifier of the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size per device",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e_composer = ExperimentComposer()

    model_path = Path(args.model_path)

    e_model = e_composer.prepare_model(
        args.model_type,
        teacher_from=model_path / "teacher_config",
        student_from=model_path / "student_config",
        use_swapnet=True,
    )

    swapnet = e_model.swapnet

    distiller = FeatureDistiller(swapnet=swapnet)
    load_model(distiller, model_path / "model.safetensors")

    swapnet.to(device)
    swapnet.eval()

    e_dset = e_composer.prepare_data(
        args.data_type,
        use_train=False,
        use_eval=True,
    )

    eval_ds = e_dset.eval

    def collate_fn(batch):
        pixels = torch.stack(
            [torch.as_tensor(example["pixel_values"]) for example in batch], dim=0
        )
        labels = torch.tensor(
            [example["labels"] for example in batch], dtype=torch.long
        )
        return {"pixel_values": pixels, "labels": labels}

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

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
