import argparse
import os

import torch
from safetensors.torch import load_file as safe_load

from pwl_model.lenet5 import LeNet5Config, LeNet5ForImageClassification


def parse_args():

    parser = argparse.ArgumentParser(
        description="Extract student weights from a distiller checkpoint"
    )
    parser.add_argument(
        "--distiller_ckpt",
        default="./ckpts/lenet-cifar10/students/checkpoint-46890",
        help="Path to the folder (or file) containing distiller pytorch_model.bin",
    )
    parser.add_argument(
        "--out_dir",
        default="./ckpts/lenet-cifar10/students/converted_student",
        help="Directory where the student-only checkpoint will be saved",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=10,
        help="Number of output classes in the student model",
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # 1) Load the distiller state dict
    sd_path = (
        args.distiller_ckpt
        if args.distiller_ckpt.endswith(".safetensors")
        else os.path.join(args.distiller_ckpt, "model.safetensors")
    )
    print(f"Loading distiller state from {sd_path} …")
    distiller_sd = safe_load(sd_path, device="cpu")

    # 2) Filter out only the student parameters
    student_sd = {
        k[len("student.") :]: v
        for k, v in distiller_sd.items()
        if k.startswith("student.")
    }
    if not student_sd:
        raise ValueError("No keys found with prefix 'student.' – check your checkpoint")

    # 3) Instantiate a fresh student
    config = LeNet5Config(num_labels=args.num_labels)
    student = LeNet5ForImageClassification(config)

    # 4) Load the filtered weights
    missing, unexpected = student.load_state_dict(student_sd, strict=False)
    print(f"→ Missing keys when loading student:   {missing}")
    print(f"→ Unexpected keys when loading student:{unexpected}")

    # 5) Save out the student-only checkpoint
    print(f"Saving student-only model to {args.out_dir} …")
    student.save_pretrained(args.out_dir)
    print("Done!")


if __name__ == "__main__":
    main()
