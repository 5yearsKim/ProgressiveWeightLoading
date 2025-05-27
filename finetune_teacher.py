import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Trainer, TrainingArguments

from pwl_model.lab import ExperimentComposer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BlockNet teacher model with MLflow integration"
    )

    parser.add_argument(
        "--model_type",
        choices=["vit"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--data_type",
        choices=["cifar10", "cifar100"],
        help="dataset to use for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=64,
        help="Batch size per device",
    )
    parser.add_argument(
        "--is_sample",
        action="store_true",
        help="Run training on a small sample of the dataset",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--teacher_config_path",
        type=str,
        required=True,
        help="Directory to save model config.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="run 1 time evaluation only",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e_composer = ExperimentComposer(
        model_type=args.model_type, dataset_name=args.data_type
    )

    e_dset = e_composer.prepare_data()

    e_model = e_composer.prepare_model(
        teacher_from=args.teacher_config_path,
        student_from=None,
        use_swapnet=False,
    )

    model = e_model.teacher.to(device)

    pretrained_path = Path(args.pretrained_path)
    block_sd = load_file(pretrained_path / "model.safetensors")

    for k in ("classifier.weight", "classifier.bias"):
        if k in block_sd:
            block_sd.pop(k)

    loading = model.load_state_dict(block_sd, strict=False)
    print("  missing keys:", loading.missing_keys)
    print("  unexpected keys:", loading.unexpected_keys)

    ds_train = e_dset.train
    ds_eval = e_dset.eval
    collate_fn = e_dset.collate_fn

    if args.is_sample:
        ds_train = ds_train.select(range(500))
        ds_eval = ds_eval.select(range(100))

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": (preds == p.label_ids).mean()}

    # training args
    training_args = TrainingArguments(
        output_dir=args.save_path,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        dataloader_num_workers=8,
    )

    # use the default collator (just stacks your tensors into batches)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    if args.eval_only:
        # just run evaluation one time and exit
        metrics = trainer.evaluate()
        print(f"*** Eval metrics: {metrics} ***")
        return

    trainer.train()


if __name__ == "__main__":
    main()
