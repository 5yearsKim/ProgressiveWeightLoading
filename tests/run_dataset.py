from datasets import load_dataset

# # for training split (streaming=True can help avoid downloading all ~150 GB at once)
# train_ds = load_dataset(
#     "mlx-vision/imagenet-1k",
#     split="train",
#     use_auth_token=True,
#     streaming=True,
# )
# for validation split
ds = load_dataset(
    # "mlx-vision/imagenet-1k",
    "imagenet-1k"
)

print(ds)