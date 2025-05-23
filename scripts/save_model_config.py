import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save a model configuration for specified model and dataset types."
    )
    parser.add_argument(
        "-m", "--model-type",
        type=str,
        required=True,
        choices=["resnet-teacher", "resnet-student", "lenet5-teacher", "lenet5-student", "vgg-teacher", "vgg-student"],
        help="Model type: teacher/student variant of ResNet or LeNet5",
    )
    parser.add_argument(
        "-d", "--data-type",
        type=str,
        required=True,
        choices=["cifar100", "cifar10"],
        help="Dataset type: CIFAR-100 or CIFAR-10",
    )
    return parser.parse_args()


def get_save_path(model_type: str, data_type: str) -> str:
    # e.g. model_type "resnet-teacher" -> base "resnet", suffix "teacher"
    base_name, variant = model_type.split("-")
    save_base = f"./ckpts/{base_name}-{data_type}/config"
    # teacher -> teacher_config, student -> student_config
    filename = f"{variant}_config"
    return os.path.join(save_base, filename)


def save_config(model_type: str, data_type: str, save_path: str) -> None:
    # determine number of labels
    if data_type == "cifar100":
        num_labels = 100
    else:
        num_labels = 10

    os.makedirs(save_path, exist_ok=True)

    if model_type == "resnet-teacher":
        from pwl_model.models.resnet import BlockResNetConfig

        config = BlockResNetConfig.from_pretrained("microsoft/resnet-18")
        # teacher-specific settings
        config.embedder_kernel_size = 3
        config.embedder_kernel_stride = 1
        config.embedder_use_pooler = False
        config.downsample_in_first_stage = False
        config.num_labels = num_labels
        config.save_pretrained(save_path)

    elif model_type == "resnet-student":
        from pwl_model.models.resnet import BlockResNetConfig

        config = BlockResNetConfig.from_pretrained("microsoft/resnet-18")
        # student-specific settings
        config.embedder_kernel_size = 3
        config.embedder_kernel_stride = 1
        config.embedder_use_pooler = False
        config.downsample_in_first_stage = False
        config.num_labels = num_labels
        config.hidden_sizes = [32, 64, 128, 256]
        config.save_pretrained(save_path)

    if model_type == "lenet5-teacher":
        from pwl_model.models.lenet5 import BlockLeNet5Config

        config = BlockLeNet5Config(
            cnn_channels=[6, 16],
            fc_sizes=[400, 120, 84],
            num_labels=num_labels,
        )

        config.save_pretrained(save_path)

    elif model_type == "lenet5-stdent":
        from pwl_model.models.lenet5 import BlockLeNet5Config
        # lenet5 config doesn't need pretrained, just instantiate
        config = BlockLeNet5Config(cnn_channels=[3, 8], fc_sizes=[200, 120, 84], num_labels=num_labels)
        config.save_pretrained(save_path)
    elif model_type == "vgg-teacher":
        from pwl_model.models.vgg import BlockVGGConfig

        config = BlockVGGConfig(
            hidden_sizes=[64, 128, 256, 512],
            depths=[2, 2, 3, 3],
            num_labels=num_labels,
        )
        config.save_pretrained(save_path)
    elif model_type == "vgg-student":
        from pwl_model.models.vgg import BlockVGGConfig

        config = BlockVGGConfig(
            hidden_sizes=[16, 32, 64, 128],
            depths=[2, 2, 3, 3],
            num_labels=num_labels,
        )
        config.save_pretrained(save_path)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()
    save_path = get_save_path(args.model_type, args.data_type)
    save_config(args.model_type, args.data_type, save_path)


if __name__ == "__main__":
    main()