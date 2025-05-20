from transformers import AutoModelForImageClassification, AutoImageProcessor
from pwl_model.models.resnet import BlockResNetForImageClassification, convert_hf_to_block_resnet


SAVE_PATH = './ckpts/resnet/teacher/ms_resnet_18'

hf_resnet = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
hf_image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

hf_state_dict = hf_resnet.state_dict()
block_state_dict = convert_hf_to_block_resnet(hf_state_dict)

config = hf_resnet.config
block_resnet = BlockResNetForImageClassification(config)
block_resnet.load_state_dict(block_state_dict, strict=False)

block_resnet.save_pretrained(SAVE_PATH)
hf_image_processor.save_pretrained(SAVE_PATH)

