
MODEL_TYPE = "resnet" 

if MODEL_TYPE == "resnet":
    SAVE_PATH = "./ckpts/resnet/student/base_config"
elif MODEL_TYPE == "lenet5":
    SAVE_PATH = "./ckpts/lenet-cifar10/students/base_config"
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

def save_student_config():
    if MODEL_TYPE == "resnet":
        from pwl_model.models.resnet import BlockResNetConfig
        student_config = BlockResNetConfig.from_pretrained("microsoft/resnet-18")
        student_config.hidden_sizes = [32, 64, 128, 256]
        student_config.save_pretrained(SAVE_PATH)

    elif MODEL_TYPE == "lenet5":
        from pwl_model.models.lenet5 import BlockLeNet5Config
        student_config = BlockLeNet5Config(cnn_channels=[3, 8], fc_sizes=[200, 120, 84])
        student_config.save_pretrained(SAVE_PATH)

    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

if __name__ == "__main__":
    save_student_config()