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

def run_block_vit_summary():
    from pwl_model.models.vit import BlockViTConfig, BlockViTForImageClassification, BlockViTModel

    model_name = "WinKawaks/vit-tiny-patch16-224"
    config = BlockViTConfig.from_pretrained(model_name, layers_per_block=2)
       

    model = BlockViTForImageClassification(config)
    model = model.to("cpu").eval()




    x = torch.randn(1, 3, 224, 224)  # batch size of 1
    print(model(x).logits.shape)  # (1, 64, 128)
    summary(
        model,
        input_size=(1, 3, 224, 224),                    # (batch, C, H, W)
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=5,                                      # how many nested layers to show
    )



def run_compare_block_vit():
    from pwl_model.models.vit import BlockViTConfig, BlockViTForImageClassification, BlockViTModel
    from transformers import ViTConfig, ViTForImageClassification

    model_name = "WinKawaks/vit-tiny-patch16-224"
    config = BlockViTConfig.from_pretrained(model_name, layers_per_block=2)

    x = torch.zeros([1, 3, 224, 224])       

    original_model = ViTForImageClassification.from_pretrained(model_name)


    # model = BlockViTForImageClassification(config)
    model = BlockViTForImageClassification.from_pretrained("./ckpts/converted/vit-tiny")


    model.load_state_dict(original_model.state_dict(), strict=False)

    model = model.to("cpu").eval()
    original_model = original_model.to('cpu').eval()

    print('---original vit-----')
    print(original_model)
    print('----block vit----')
    print(model)

    out1 = model(x)
    out2 = original_model(x)

    print('out1 shape: ', out1.logits.shape)
    print('out2 shape: ', out2.logits.shape)

    print("Max abs diff:", (out1.logits - out2.logits).abs().max().item())



    
if __name__ == "__main__":
    # run_block_resnet_summary()
    # run_vits_summary()
    # run_block_vgg_summary()
    # run_block_vit_summary()
    run_compare_block_vit()
