import os
import argparse
from pathlib import Path

from pwl_model.lab import ExperimentComposer
from torchinfo import summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Save a model configuration for specified model and dataset types."
    )
    parser.add_argument(
        "-m", "--model-type",
        type=str,
        required=True,
        choices=[
            "resnet",
            "lenet5", 
            "vgg", 
            "vit",
        ],
        help="Model type"
    )
    parser.add_argument(
        "-d", "--data-type",
        type=str,
        required=True,
        choices=["cifar100", "cifar10"],
        help="Dataset type: CIFAR-100 or CIFAR-10",
    )
    parser.add_argument(
        "-D", "--depth",
        type=int,
	default=5,
        help="depth",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.model_type == 'vit':
        INPUT_SHAPE = (1, 3, 224, 224)
    else:
        INPUT_SHAPE = (1, 3, 32, 32)
    DEPTH=args.depth


    target_path = Path(f"./ckpts/{args.model_type}-{args.data_type}/student/checkpoint-main")


    e_composer = ExperimentComposer(
        model_type=args.model_type,
        dataset_name=args.data_type
    )

    e_model = e_composer.prepare_model(
        teacher_from=target_path / 'teacher_config',
        student_from=target_path / 'student_config',
    )

    teacher = e_model.swapnet.teacher
    student = e_model.swapnet.student


    print("---teacher---")
    summary(
        teacher,
        input_size=INPUT_SHAPE,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=DEPTH,
    )

    print('---student---')
    summary(
        student,
        input_size=INPUT_SHAPE,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=DEPTH,
    )


if __name__ == "__main__":
    main()





