from pwl_model.resnet import ResNetFeatureDistiller
from transformers import ResNetForImageClassification, ResNetConfig, Trainer, AutoImageProcessor \
    , TrainingArguments
from datasets import load_dataset

TEACHER_PATH = './ckpts/resnet/resnet18'
teacher = ResNetForImageClassification.from_pretrained(TEACHER_PATH)
teacher_config = teacher.config


student_config = ResNetConfig(**teacher_config.to_dict())
student_config.depths = [max(1, d // 2) for d in student_config.depths]

student = ResNetForImageClassification(student_config)

# 5. Preprocess: convert images → pixel_values, and get labels
#    cats-image has no labels, so we’ll “pseudo-label” via teacher’s top‐1.

preprocessor = AutoImageProcessor.from_pretrained(TEACHER_PATH)
def preprocess(batch):
    # batch["image"] is a list of PIL images
    inputs = preprocessor(batch["image"], return_tensors="pt")
    batch["pixel_values"] = inputs["pixel_values"]
    print('preprocess format:', batch["pixel_values"].shape)
    return batch


train_ds, val_ds = load_dataset("zh-plus/tiny-imagenet", split=["train[:100]", "valid[:100]"])
train_ds = train_ds.map(preprocess, 
    batched=True,
    batch_size=32, # not for the training loop.. only for the dataset
    remove_columns=["image"], 
)
train_ds.set_format(type="torch", columns=["pixel_values", "label"])


# for item in train_ds:
#     print(item.keys())
#     print(item['pixel_values'].shape)
#     print(item['label'])
#     break


distiller = ResNetFeatureDistiller(student, teacher)


# 7. Define a Trainer that uses our distiller
class DistilTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs:bool=False, **kwargs):
        loss_dict = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
        )
        loss = loss_dict["loss"]
        return (loss, loss_dict) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./cats-distill",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=50,
    save_steps=500,
)

trainer = DistilTrainer(
    model=distiller,
    args=training_args,
    train_dataset=train_ds,
)

# 8. Kick off training!
trainer.train()
