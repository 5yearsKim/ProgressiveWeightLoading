from datasets import load_dataset

# # for training split (streaming=True can help avoid downloading all ~150 GB at once)
# train_ds = load_dataset(
#     "mlx-vision/imagenet-1k",
#     split="train",
#     use_auth_token=True,
#     streaming=True,
# )
# for validation split
val_ds = load_dataset(
    "mlx-vision/imagenet-1k",
    split="validation",
    use_auth_token=True,
    streaming=True,
)

item = next(val_ds)

print(item)