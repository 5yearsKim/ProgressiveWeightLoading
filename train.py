import argparse

from pwl_model.resnet import ResNetFeatureDistiller
from transformers import (
    ResNetForImageClassification,
    ResNetConfig,
    Trainer,
    AutoImageProcessor,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a ResNet teacher into a smaller ResNet student."
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        default="./ckpts/resnet/resnet18",
        help="Path or model identifier of the pretrained teacher",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="zh-plus/tiny-imagenet",
        help="🤗 dataset identifier (e.g. 'zh-plus/tiny-imagenet')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ckpts/resnet/resnet18_distill",
        help="Where to save distilled checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    return parser.parse_args()


class DistilTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # standard HuggingFace Trainer calls model(...) under the hood
        loss_dict = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
        )
        loss = loss_dict["loss"]
        return (loss, loss_dict) if return_outputs else loss


def main():
    args = parse_args()

    # 1. Load teacher & build student config
    teacher = ResNetForImageClassification.from_pretrained(args.teacher_path)
    teacher_cfg = teacher.config

    student_cfg = ResNetConfig(**teacher_cfg.to_dict())
    # halve the depths of each stage
    student_cfg.depths = [max(1, d // 2) for d in student_cfg.depths]
    student = ResNetForImageClassification(student_cfg)

    # 2. Preprocessor (to get pixel_values)
    processor = AutoImageProcessor.from_pretrained(args.teacher_path)

    def preprocess(batch):
        inputs = processor(batch["image"], return_tensors="pt")
        batch["pixel_values"] = inputs["pixel_values"]
        return batch

    # 3. Load & preprocess data
    train_ds, val_ds = load_dataset(
        args.dataset_name, split=["train[:100]", "valid[:100]"]
    )
    train_ds = train_ds.map(
        preprocess,
        batched=True,
        batch_size=32,
        remove_columns=["image"],
    )
    # rename for Trainer
    train_ds = train_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    # 4. Prepare distiller
    distiller = ResNetFeatureDistiller(student, teacher)

    # 5. TrainingArguments via argparse
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        save_safetensors=True,
        report_to=["mlflow"],
        logging_dir="./runs",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
    )

    # 6. Trainer + MLflow callback
    trainer = DistilTrainer(
        model=distiller,
        args=training_args,
        train_dataset=train_ds,
        callbacks=[MLflowCallback()],
    )

    # 7. Launch!
    trainer.train()


if __name__ == "__main__":
    main()
