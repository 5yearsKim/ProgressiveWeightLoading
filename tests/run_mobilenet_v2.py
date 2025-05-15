from transformers import MobileNetV2ForImageClassification, MobileNetV2ImageProcessor
from datasets     import load_dataset, DownloadMode
from PIL          import Image
import torch

from torchinfo import summary

model_name = "AiresPucrs/Mobilenet-v2-CIFAR-10"

# 1) Load processor & model, send to device
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = MobileNetV2ImageProcessor.from_pretrained(model_name)
model     = MobileNetV2ForImageClassification.from_pretrained(model_name).to(device)
model.eval()

# 2) Load CIFAR-10 test split
ds_test = load_dataset("uoft-cs/cifar10", split="test")



# 3) Iterate in batches and accumulate correct / total
BATCH_SIZE = 32
correct, total = 0, 0

for i in range(0, len(ds_test), BATCH_SIZE):
    batch = ds_test[i : i + BATCH_SIZE]
    # a) Convert each image (H×W×C numpy array) to PIL



    imgs = [img.convert("RGB") for img in batch["img"]]
    # b) Preprocess
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    # c) Inference
    with torch.no_grad():
        outputs = model(**inputs)
        preds   = outputs.logits.argmax(dim=-1).cpu().tolist()
    # d) Compare to ground-truth labels
    labels = batch["label"]
    correct += sum(p == l for p, l in zip(preds, labels))
    total   += len(labels)

    if i > 50:
        break

# 4) Compute accuracy
accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")
