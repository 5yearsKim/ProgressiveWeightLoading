import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

teacher_path = "./ckpts/resnet/resnet18"

image_processor = AutoImageProcessor.from_pretrained(teacher_path)
# model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

model = ResNetForImageClassification.from_pretrained(teacher_path)

inputs = image_processor(image, return_tensors="pt")

print("inputs")
print(inputs["pixel_values"].shape)

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
