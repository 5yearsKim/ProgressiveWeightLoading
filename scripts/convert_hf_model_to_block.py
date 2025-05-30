import os
import argparse
from transformers import AutoModelForImageClassification, AutoImageProcessor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace ResNet to BlockResNet and save locally."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "vit"],
        # default="microsoft/resnet-18",
        help="model type",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        # default="microsoft/resnet-18",
        help="HuggingFace model ID or local checkpoint directory",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        # default="./ckpts/resnet/teacher/ms_resnet_18",
        help="Directory where the converted model will be saved",
    )
    parser.add_argument(
        "--save_image_processor",
        action="store_true",
        help="Also save the HF image processor alongside the model",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Load the HF model
    hf_model = AutoModelForImageClassification.from_pretrained(args.pretrained_path)

    # 2) Convert its state dict to BlockResNet format
    hf_state_dict = hf_model.state_dict()

    if args.model_type == 'resnet':
        from pwl_model.models.resnet import (
            BlockResNetForImageClassification,
            convert_hf_to_block_resnet,
            check_weight_same_resnet,
        )
        block_state_dict = convert_hf_to_block_resnet(hf_state_dict)

        config = hf_model.config
        block_model = BlockResNetForImageClassification(config)
        block_model.load_state_dict(block_state_dict, strict=False)


        check_weight_same_resnet(block_model.state_dict(), hf_model.state_dict())

    elif args.model_type == 'vit':
        from pwl_model.models.vit import (
            BlockViTForImageClassification,
            convert_hf_to_block_vit,
            check_weight_same_vit,
        )

        block_state_dict = convert_hf_to_block_vit(hf_state_dict)

        config = hf_model.config
        config.layers_per_block = 2

        block_model = BlockViTForImageClassification(config)
        block_model.load_state_dict(block_state_dict)

        check_weight_same_vit(block_model.state_dict(), hf_model.state_dict())

    else:
        raise ValueError(f'no model_type {args.model_type}')
    


    os.makedirs(args.save_path, exist_ok=True)
    block_model.save_pretrained(args.save_path)

    if args.save_image_processor:
        hf_image_processor = AutoImageProcessor.from_pretrained(args.pretrained_path)
        hf_image_processor.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()
