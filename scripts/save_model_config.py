
MODEL_TYPE = "resnet-teacher" 


if MODEL_TYPE == "resnet-teacher":
    SAVE_PATH = "./ckpts/resnet-cifar100/config/teacher_config"
elif MODEL_TYPE == 'resnet-student':
    SAVE_PATH = "./ckpts/resnet-cifar100/config/student_config"
elif MODEL_TYPE == "lenet5":
    SAVE_PATH = "./ckpts/lenet-cifar10/students/base_config"
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

def save_config():
    if MODEL_TYPE == "resnet-teacher":
        from pwl_model.models.resnet import BlockResNetConfig
        config = BlockResNetConfig.from_pretrained("microsoft/resnet-18")
        config.embedder_kernel_size = 5
        config.embedder_kernel_stride = 1
        config.embedder_use_pooler = False
        config.downsample_in_first_stage = False
        config.num_labels = 100
        config.save_pretrained(SAVE_PATH)

    elif MODEL_TYPE == "resnet-student":
        from pwl_model.models.resnet import BlockResNetConfig
        config = BlockResNetConfig.from_pretrained("microsoft/resnet-18")
        config.hidden_sizes = [32, 64, 128, 256]
        config.save_pretrained(SAVE_PATH)

    elif MODEL_TYPE == "lenet5":
        from pwl_model.models.lenet5 import BlockLeNet5Config
        config = BlockLeNet5Config(cnn_channels=[3, 8], fc_sizes=[200, 120, 84])
        config.save_pretrained(SAVE_PATH)

    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

if __name__ == "__main__":
    save_config()