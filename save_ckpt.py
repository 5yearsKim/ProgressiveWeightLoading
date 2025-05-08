from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

# Load the processor and model
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

# Define the local directory to save the checkpoint
local_dir = "./ckpts/resnet/resnet18"

# Save the processor and model to the local directory
image_processor.save_pretrained(local_dir)
model.save_pretrained(local_dir)


print(f"Checkpoint saved to: {os.path.abspath(local_dir)}")

