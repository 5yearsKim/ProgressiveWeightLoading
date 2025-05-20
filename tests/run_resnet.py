import torch
from pwl_model.models.resnet import BlockResNetConfig, BlockResNetModel, BlockResNetForImageClassification
from torchsummary import summary



block_config = BlockResNetConfig.from_pretrained("microsoft/resnet-18")
# model = BlockResNetModel(config)
block_model = BlockResNetForImageClassification(block_config)


def forward_fn(model, x):
    pixel_value = torch.zeros([1, 3, 224, 224])

    out = model(pixel_value)

    print(out.logits.shape)

def print_state_dict(model):
    sd = model.state_dict()
    for name, tensor in sd.items():
        print(f"{name:30s} {tuple(tensor.shape)}")

def compare_original_resnet():
    from transformers import AutoModelForImageClassification

    ms_resnet = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

    print ('-------block model-------')
    print_state_dict(block_model)
    print ('-------ms model-------')
    print_state_dict(ms_resnet)




if __name__ == "__main__":
    compare_original_resnet()
