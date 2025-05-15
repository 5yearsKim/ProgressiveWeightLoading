import mlflow
import torch
import torch.nn as nn
from datasets import load_dataset
from torchvision import transforms as T
from transformers import Trainer, TrainingArguments
from transformers.integrations import MLflowCallback

from pwl_model.lenet5 import LeNet5Config, LeNet5ForImageClassification

RESUME_FROM_CHECKPOINT = False
EPOCHS = 20
LEARNING_RATE = 2e-3
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_experiment("lenet5-cifar10")
mlflow.log_param("device", str(device))

config = LeNet5Config()
model = LeNet5ForImageClassification(config).to(device)

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


# accuracy metric
import numpy as np


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}


# training args
training_args = TrainingArguments(
    output_dir="./ckpts/lenet-cifar10",
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
