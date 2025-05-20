import argparse

import mlflow
import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from transformers.integrations import MLflowCallback

from pwl_experiments import prepare_experiment
from pwl_model.models.lenet5 import BlockLeNet5Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BlockNet teacher model with MLflow integration"
    )

    parser.add_argument(
        "--model_type",
        choices=["lenet5", "resnet"],
        default="lenet5",
        help="Type of model to train: lenet5 or resnet",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-3,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size per device",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from latest checkpoint",
    )
    parser.add_argument(
        "--is_sample",
        action="store_true",
        help="Run training on a small sample of the dataset",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="lenet5-cifar10-teacher",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./ckpts/lenet-cifar10/teachers",
        help="Directory to save model checkpoints",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MLflow configuration
    mlflow.set_experiment(args.experiment_name)
    mlflow.log_param("device", str(device))
    mlflow.log_param("model_type", args.model_type)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("batch_size", args.batch_size)

    # prepare configuration and experiment
    if args.model_type == "lenet5":
        config = BlockLeNet5Config()
        teacher_from = config
        student_from = None
    elif args.model_type == "resnet":
        raise NotImplementedError("ResNet model type is not implemented yet.")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    e_set = prepare_experiment(
        args.model_type,
        teacher_from=teacher_from,
        student_from=student_from,
        use_swapnet=False,
    )

    model = e_set.teacher.to(device)
    ds_train = e_set.dataset.train
    ds_eval = e_set.dataset.eval

    if args.is_sample:
        ds_train = ds_train.select(range(500))
        ds_eval = ds_eval.select(range(100))

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": (preds == p.label_ids).mean()}

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.save_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_strategy="epoch",
        eval_strategy="epoch",
        report_to=["mlflow"],
        logging_dir="./runs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,
    )

    print("Training start...")

    # Trainer instantiation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback()],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
