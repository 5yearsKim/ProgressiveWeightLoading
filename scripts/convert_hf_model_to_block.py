from transformers import AutoModelForImageClassification, AutoImageProcessor
from pwl_model.models.resnet import BlockResNetForImageClassification, convert_hf_to_block_resnet, check_weight_same_resnet 


SAVE_PATH = './ckpts/resnet/teacher/ms_resnet_18'
PRETRAINED_PATH = "microsoft/resnet-18"
SAVE_IMAGE_PROCESSOR=False

def convert():
    hf_model = AutoModelForImageClassification.from_pretrained(PRETRAINED_PATH)

    hf_state_dict = hf_model.state_dict()
    block_state_dict = convert_hf_to_block_resnet(hf_state_dict)

    config = hf_model.config
    block_model = BlockResNetForImageClassification(config)
    block_model.load_state_dict(block_state_dict, strict=False)

    check_weight_same_resnet(block_model.state_dict(), hf_model.state_dict())

    block_model.save_pretrained(SAVE_PATH)
    if SAVE_IMAGE_PROCESSOR:
        hf_image_processor = AutoImageProcessor.from_pretrained(PRETRAINED_PATH)
        hf_image_processor.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    convert()
