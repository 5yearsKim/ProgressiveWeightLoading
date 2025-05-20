from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn as nn 
from torchsummary import summary
# from torchinfo import summary
from datasets import load_dataset

SAVE_PATH: None|str = None

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]


image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")


if SAVE_PATH:
    image_processor.save_pretrained(SAVE_PATH)
    model.save_pretrained(SAVE_PATH)

# inputs = image_processor(image, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])

model.eval()

# Wrap so that we pull out the first element of the tuple
def forward_fn(x):
    out = model(x, return_dict=False).logits
    return out            # the tensor of shape [B, C, H, W]

class WrappedModel(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        # return_dict=False → tuple; [0] is the feature tensor
        out_tuple = self.hf_model(x, return_dict=False)
        return out_tuple

wrapped = WrappedModel(model)

# 3) Now this will work:
summary(wrapped, input_size=(3, 224, 224), device="cpu")

