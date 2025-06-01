import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.optim as optim
from safetensors.torch import load_file
from transformers import (Trainer, TrainerCallback, TrainerControl,
                          TrainerState, TrainingArguments)
from transformers.integrations import MLflowCallback
from transformers.trainer_utils import EvalPrediction

from pwl_model.core.feature_distiller import DistillerOutput, FeatureDistiller
from pwl_model.lab import ExperimentComposer
from pwl_model.utils.training_utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a teacher into a smaller student."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "lenet5", "vgg", "vit"],
        help="Model type",
    )
    parser.add_argument(
        "--data_type",
        choices=["cifar10", "cifar100", "imagenet"],
        help="dataset to use for training",
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        help="Path or model identifier of the pretrained teacher",
    )
    parser.add_argument(
        "--student_path",
        type=str,
        help="Path for the student config",
    )
    parser.add_argument(
        "--student_pretrained_path",
        type=str,
        default=None,
        help="Path for the student pretrained weight path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Where to save distilled checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=160,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-2,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="minimum learning rate",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=128,
        help="Batch size per device",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="If set, resume training from the last checkpoint",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--is_sample",
        action="store_true",
        help="Take only sample dataset for the development purpose",
    )
    parser.add_argument(
        "--cross_mode",
        type=str,
        choices=["random", "all"],
        default="random",
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

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)

        swapnet = self.model.swapnet

        # student
        student_cfg = swapnet.student.config

        student_dir = os.path.join(output_dir, "student_config")
        os.makedirs(student_dir, exist_ok=True)
        student_cfg.save_pretrained(student_dir)

        # teacher
        teacher_cfg = swapnet.teacher.config

        teacher_dir = os.path.join(output_dir, "teacher_config")
        os.makedirs(teacher_dir, exist_ok=True)
        teacher_cfg.save_pretrained(teacher_dir)


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

        print("----------------------")
        print(logs)

        # Tell HF to write these logs now
        control.should_log = True
        return control


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_experiment(args.experiment_name)
    mlflow.log_param("device", str(device))
    mlflow.log_param("model_type", args.model_type)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("learning_rate", args.lr)
    mlflow.log_param("batch_size", args.bs)

    e_composer = ExperimentComposer(
        model_type=args.model_type, dataset_name=args.data_type
    )

    e_model = e_composer.prepare_model(
        teacher_from=args.teacher_path,
        student_from=args.student_path,
    )

    e_dset = e_composer.prepare_data()

    swapnet = e_model.swapnet

    if args.student_pretrained_path:

        pretrained_path = Path(args.student_pretrained_path)
        print("Loading student weight from ", pretrained_path)
        block_sd = load_file(pretrained_path / "model.safetensors")

        # 2) Extract only keys that start with "swapnet.student.", stripping that prefix:
        prefix = "swapnet.student."
        student_sd = {}
        for full_key, tensor in block_sd.items():
            if full_key.startswith(prefix):
                new_key = full_key[len(prefix) :]
                student_sd[new_key] = tensor

        for k in ("classifier.weight", "classifier.bias"):
            if k in student_sd:
                student_sd.pop(k)

        loading = swapnet.student.load_state_dict(student_sd, strict=False)
        print("  missing keys:", loading.missing_keys)
        print("  unexpected keys:", loading.unexpected_keys)



    ds_train = e_dset.train
    ds_eval = e_dset.eval
    collate_fn = e_dset.collate_fn

    optimizer = optim.SGD(
        [
            {"params": swapnet.encoders.parameters(), "lr": args.lr * 0.2},
            {"params": swapnet.decoders.parameters(), "lr": args.lr * 0.2},
            {
                "params": (
                    p
                    for name, p in swapnet.named_parameters()
                    if p.requires_grad
                    and not name.startswith("encoders")
                    and not name.startswith("decoders")
                ),
                "lr": args.lr,
            },
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )

    epoch_steps = len(ds_train) // args.bs

    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epoch_steps * args.epochs, eta_min=min(args.lr / 20, args.min_lr)
    )

    if args.is_sample:
        ds_train = ds_train.select(range(500))
        ds_eval = ds_eval.select(range(100))

    distiller = FeatureDistiller(
        swapnet=swapnet,
        cross_mode=args.cross_mode,
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
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_safetensors=True,
        report_to=["mlflow"],
        logging_dir="./runs",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=6,
    )

    trainer = DistilTrainer(
        model=distiller,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback(), MeterCallback()],
        optimizers=(optimizer, train_scheduler),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
