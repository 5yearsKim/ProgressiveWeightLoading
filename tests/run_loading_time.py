import time
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Measure model loading and inference time")
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        choices=["vgg", "resnet", "vit"],
        help="Model type to test (vgg, resnet, vit)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_type = args.model

    if model_type == 'vgg':
        from pwl_model.models.vgg import BlockVGGForImageClassification
        ckpt_path = "./ckpts/vgg-cifar10/teacher/checkpoint-main"
        input_shape = (1, 3, 32, 32)
        BaseModel = BlockVGGForImageClassification

    elif model_type == 'resnet':
        from pwl_model.models.resnet import BlockResNetForImageClassification
        ckpt_path = "./ckpts/resnet-cifar10/teacher/checkpoint-main"
        input_shape = (1, 3, 32, 32)
        BaseModel = BlockResNetForImageClassification

    elif model_type == 'vit':
        from pwl_model.models.vit import BlockViTForImageClassification
        ckpt_path = "./ckpts/vit-cifar10/teacher/checkpoint-main"
        input_shape = (1, 3, 224, 224)
        BaseModel = BlockViTForImageClassification

    else:
        raise ValueError("Unsupported model type")

    print(f'Loading model from {ckpt_path}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Measure model loading time
    start_time = time.time()
    model = BaseModel.from_pretrained(ckpt_path)
    model = model.to(device)
    end_time = time.time()
    load_duration_ms = (end_time - start_time) * 1000
    print(f"Model loaded in {load_duration_ms:.2f} ms")

    # Check device
    first_param_device = next(model.parameters()).device
    print(f"Model is on device: {first_param_device}")

    # Prepare dummy input
    dummy_input = torch.randn(*input_shape).to(device)

    # Measure inference time
    model.eval()
    with torch.no_grad():
        start_inf = time.time()
        _ = model(dummy_input)
        end_inf = time.time()

    inference_duration_ms = (end_inf - start_inf) * 1000
    print(f"First inference took {inference_duration_ms:.2f} ms")


if __name__ == "__main__":
    main()
