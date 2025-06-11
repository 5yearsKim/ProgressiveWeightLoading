import os
import math
import torch
import torch.nn.functional as F
from transformers import ViTModel, ViTImageProcessor
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size':        24,   # base font size for text
    'axes.titlesize':   24,   # subplot title
    'axes.labelsize':   24,   # x/y labels
    'xtick.labelsize':  24,   # tick labels
    'ytick.labelsize':  24,
    'legend.fontsize':  20,
})

# 1) Configuration
transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Pick three fixed samples by index
sample_indices = [1, 4, 6]

# Two models to compare: “student” vs “teacher”
model_ids = [
    'google/vit-base-patch16-224',
    'google/vit-base-patch16-224-in21k'
]

# Layers to visualize (1, 3, 5, 7 → zero‐based indices 0,2,4,6)
layer_indices = [0, 2, 4, 6]
layer_labels = ['block1', 'bock2', 'block3', 'block4']

# 2) Load both ViT models (attentions enabled) and a shared processor
processor = ViTImageProcessor.from_pretrained(model_ids[0])
models = [
    ViTModel.from_pretrained(m_id, output_attentions=True).eval()
    for m_id in model_ids
]

# 3) Prepare the canvas: each sample uses 2 rows (student/teacher) and 1+4 columns
rows = len(sample_indices) * 2
cols = 1 + len(layer_indices)
fig, axes = plt.subplots(
    rows, cols,
    figsize=(3 * cols, 3 * rows),
    gridspec_kw={'hspace': 0.1}  # add vertical gap between blocks
)

for sample_i, idx in enumerate(sample_indices):
    img_tensor, _ = testset[idx]
    batch = img_tensor.unsqueeze(0)
    # Denormalize image
    img = img_tensor.permute(1, 2, 0).numpy()
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    img = std * img + mean
    img = (img - img.min()) / (img.max() - img.min())

    for model_i, (model_id, model) in enumerate(zip(model_ids, models)):
        row = sample_i * 2 + model_i
        role = 'Student' if model_i == 0 else 'Teacher'

        # Column 0: original image only for Student row
        ax = axes[row, 0]
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)


        if model_i == 0:
            ax.imshow(img)
            ax.set_title('Original')


        ax.set_ylabel(role, rotation=0, labelpad=60, va='center')

        # Forward pass to get attentions
        with torch.no_grad():
            out = model(batch)
        attns = out.attentions

        # Columns 1–4: attention overlays
        for j, layer_idx in enumerate(layer_indices):
            ax = axes[row, j + 1]
            avg_heads = attns[layer_idx][0].mean(dim=0)
            cls_attn = avg_heads[0, 1:].cpu()
            P = int(math.sqrt(cls_attn.shape[0]))
            attn_map = cls_attn.reshape(1, 1, P, P)
            attn_map = F.interpolate(
                attn_map, size=(224, 224),
                mode='bilinear', align_corners=False
            )[0, 0].numpy()

            ax.imshow(img)
            ax.imshow(attn_map, cmap='jet', alpha=0.5)
            ax.axis('off')
            # Only label the layer titles on the first Student row
            if model_i == 0:
                ax.set_title(layer_labels[j])


  
plt.subplots_adjust(top=0.97, bottom=0.02, hspace=0.1)  

plt.savefig('data/result/attention_map.png', format='png')
