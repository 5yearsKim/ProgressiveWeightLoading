import argparse

import mlflow
import numpy as np
import torch
import torch.optim as optim
from transformers import Trainer, TrainingArguments, SchedulerType
from transformers.integrations import MLflowCallback

from pwl_model.lab import prepare_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BlockNet teacher model with MLflow integration"
    )

    parser.add_argument(
        "--model_type",
        choices=["lenet5", "resnet"],
        help="Type of model to train: lenet5 or resnet",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-3,
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
        help="MLflow experiment name",
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

    # MLflow configuration
    mlflow.set_experiment(args.experiment_name)
    mlflow.log_param("device", str(device))
    mlflow.log_param("model_type", args.model_type)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("batch_size", args.batch_size)

    # prepare configuration and experiment
    teacher_from = args.pretrained_path

    e_set = prepare_experiment(
        args.model_type,
        teacher_from=teacher_from,
        student_from=None,
        use_swapnet=False,
    )

    model = e_set.teacher.to(device)
    ds_train = e_set.dataset.train
    ds_eval = e_set.dataset.eval
    collate_fn = e_set.dataset.collate_fn

    # optimizer = optim.SGD(
    #     model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    # )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    epoch_steps = 50000 // args.batch_size

    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_steps * args.epochs, eta_min=args.learning_rate/20) #learning rate decay

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
        warmup_steps=500,
        # lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
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
        dataloader_num_workers=8,
    )

    print("Training start...")

    # Trainer instantiation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback()],
        optimizers=(optimizer, train_scheduler),
    )

    if args.eval_only:
        # just run evaluation one time and exit
        metrics = trainer.evaluate()
        print(f"*** Eval metrics: {metrics} ***")
        return

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
