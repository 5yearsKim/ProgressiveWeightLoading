import torch
from pwl_model.models.resnet import BlockResNetConfig, BlockResNetForImageClassification
from torchinfo import summary

def run_block_resnet_summary():
    # model = BlockResNetModel(config)
    config = BlockResNetConfig.from_pretrained("./ckpts/resnet-cifar100/config/teacher_config")
    model = BlockResNetForImageClassification(config)

    model = model.to('cpu').eval()

    summary(
        model,
        input_size=(1, 3, 32, 32),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=5,           # how many nested layers to show
    )


def run_block_vgg_summary():
    from pwl_model.models.vgg import BlockVGGConfig, BlockVGGForImageClassification
    
    config = BlockVGGConfig(
        hidden_sizes = [64, 128, 256, 512 ],
        depths = [2, 2, 3, 3 ],
    )
    model = BlockVGGForImageClassification(config)

    model = model.to('cpu').eval()

    summary(
        model,
        input_size=(1, 3, 32, 32),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=5,           # how many nested layers to show
    )

def run_vits_summary():
    from transformers import ViTConfig, ViTForImageClassification

    config = ViTConfig(
        image_size=32,             # CIFAR-10 images are 32×32
        patch_size=4,              # 4×4 patches → (32/4)^2 = 64 tokens
        num_channels=3,            # RGB
        hidden_size=128,           # token dimension
        num_hidden_layers=8,       # depth
        num_attention_heads=8,     # heads
        intermediate_size=1024,    # MLP hidden dim = 4×hidden_size
        num_labels=10,             # CIFAR-10 has 10 classes
    )

    # 2) Instantiate the model
    model = ViTForImageClassification(config)
    model = model.to("cpu").eval()

    # 3) Print a summary
    summary(
        model,
        input_size=(1, 3, 32, 32),                    # (batch, C, H, W)
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=5,                                      # how many nested layers to show
    )

    
if __name__ == "__main__":
    # run_block_resnet_summary()
    # run_vits_summary()
    run_block_vgg_summary()
