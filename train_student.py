import argparse
import os

import mlflow
import numpy as np
import torch
from transformers import (Trainer, TrainerCallback, TrainerControl,
                          TrainerState, TrainingArguments)
from transformers.integrations import MLflowCallback
from transformers.trainer_utils import EvalPrediction

from pwl_experiments import prepare_experiment
from pwl_model.core.feature_distiller import DistillerOutput, FeatureDistiller
from pwl_model.utils.training_utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a teacher into a smaller student."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "lenet5"],
        help="Model type",
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
        "--output_dir",
        type=str,
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

        print(logs)

        # Tell HF to write these logs now
        control.should_log = True
        return control


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "resnet":
        mlflow.set_experiment("resnet18-tiny-imagenet-distill")
        mlflow.log_param("device", str(device))

    elif args.model_type == "lenet5":
        mlflow.set_experiment("lenet5-cifar10-distill")
        mlflow.log_param("device", str(device))
    else:
        raise ValueError(f"{args.model_type} not defined")

    e_set = prepare_experiment(
        args.model_type,
        teacher_from=args.teacher_path,
        student_from=args.student_path,
    )

    swapnet = e_set.swapnet
    ds_train = e_set.dataset.train
    ds_eval = e_set.dataset.eval
    collate_fn = e_set.dataset.collate_fn

    if args.is_sample:
        ds_train = ds_train.select(range(500))
        ds_eval = ds_eval.select(range(100))

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
        eval_dataset=ds_eval,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback(), MeterCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    main()
