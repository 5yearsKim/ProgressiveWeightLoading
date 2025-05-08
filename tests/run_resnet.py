from pwl_model.resnet import ResNetFeatureDistiller
from transformers import ResNetForImageClassification, ResNetConfig

TEACHER_PATH = './ckpts/resnet/resnet18'
teacher = ResNetForImageClassification.from_pretrained(TEACHER_PATH)
teacher_config = teacher.config

_t = teacher_config
student_config = ResNetConfig(
    num_channels=_t.num_channels,
    hidden_sizes=_t.hidden_sizes,
    embedding_size=_t.embedding_size,
    depths=[max(1, d // 2) for d in _t.depths],          
    layer_type=_t.layer_type,
    hidden_act=_t.hidden_act,
    downsample_in_first_stage=_t.downsample_in_first_stage,
    downsample_in_bottleneck=_t.downsample_in_bottleneck,
)
student = ResNetForImageClassification(student_config)

distiller = ResNetFeatureDistiller(student, teacher)


import torch

mock_pixel_value = torch.zeros([1, 3, 480, 640])

out = distiller(mock_pixel_value)
print(out)