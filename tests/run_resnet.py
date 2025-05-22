import torch
from pwl_model.models.resnet import BlockResNetConfig, BlockResNetModel, BlockResNetForImageClassification
from torchsummary import summary



# model = BlockResNetModel(config)
block_model = BlockResNetForImageClassification.from_pretrained("./ckpts/resnet/teacher/ms_resnet_18")

from transformers import AutoModelForImageClassification
ms_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

def forward_fn(model, x):
    pixel_value = torch.zeros([1, 3, 32, 32])

    out = model(pixel_value)


    print(out.logits.shape)

def print_state_dict(model):
    print(model)
    # sd = model.state_dict()
    # for name, tensor in sd.items():
    #     print(f"{name:30s} {tuple(tensor.shape)}")

def compare_original_resnet():
    print ('-------block model-------')
    print_state_dict(block_model)
    print ('-------ms model-------')
    print_state_dict(ms_model)


def official_example():
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch
    from datasets import load_dataset

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    # model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    model = BlockResNetForImageClassification.from_pretrained("./ckpts/resnet/teacher/ms_resnet_18")

    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])


def test_difference():
    import torch

    # assume block_model and ms_model have identical .state_dict() mappings
    block_model.eval()
    ms_model.eval()

    x = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
    with torch.no_grad():
        y1 = block_model(x)
        y2 = ms_model(x)

    print("max abs diff:", (y1.logits - y2.logits).abs().max().item())


if __name__ == "__main__":
    # compare_original_resnet()
    official_example()
    # test_difference()