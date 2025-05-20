import mlflow
import torch
from datasets import load_dataset
from torchvision import transforms as T
from transformers import Trainer, TrainingArguments
from transformers.integrations import MLflowCallback

from pwl_model.models.lenet5 import (BlockLeNet5Config,
                                     BlockLeNet5ForImageClassification)

EPOCHS = 40
LEARNING_RATE = 3e-3
BATCH_SIZE = 128
RESUME_FROM_CHECKPOINT = False
IS_SAMPLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_experiment("lenet5-cifar10-teacher")
mlflow.log_param("device", str(device))


config = BlockLeNet5Config()
model = BlockLeNet5ForImageClassification(config).to(device)

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


ds = load_dataset("uoft-cs/cifar10")

print("processing train set")
ds_train = ds["train"].map(
    # ds_train = ds["train"].select(range(1000)).map(
    preprocess,
    batched=True,
    batch_size=64,
    remove_columns=["img", "label"],
)

print("processing test set")
ds_test = ds["test"].map(
    # ds_test = ds["test"].select(range(100)).map(
    preprocess,
    batched=True,
    batch_size=64,
    remove_columns=["img", "label"],
)


if IS_SAMPLE:
    ds_train = ds_train.select(range(500))
    ds_test = ds_test.select(range(100))

# accuracy metric
import numpy as np


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}


# training args
training_args = TrainingArguments(
    output_dir="./ckpts/lenet-cifar10/teacher_training",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
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

print("training start..")

# instantiate and run
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    compute_metrics=compute_metrics,
    callbacks=[MLflowCallback()],
)

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
