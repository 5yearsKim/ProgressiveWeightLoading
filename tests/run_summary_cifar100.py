import torch
from pwl_model.models.resnet import BlockResNetConfig, BlockResNetModel, BlockResNetForImageClassification
from torchinfo import summary

import torch.nn as nn



# model = BlockResNetModel(config)
config = BlockResNetConfig.from_pretrained("./ckpts/resnet-cifar100/config/teacher_config")
model = BlockResNetForImageClassification(config)


def forward_fn(x):

    out = model(x)

    return out.logits

model = model.to('cpu').eval()

summary(
    model,
    input_size=(1, 3, 32, 32),
    col_names=("input_size", "output_size", "num_params", "trainable"),
    depth=5,           # how many nested layers to show
)

