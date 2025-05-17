import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention)


class LeNet5Config(PretrainedConfig):
    model_type = "lenet5"

    def __init__(
        self,
        in_channels: int = 3,
        num_labels: int = 10,
        cnn_channels: list[int]=[6, 16],
        fc_sizes: list[int]=[400, 120, 84],
        **kwargs,
    ):

        assert len(cnn_channels) == 2
        assert len(fc_sizes) == 3
        assert cnn_channels[-1] * 5 * 5 == fc_sizes[0]

        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.num_labels = num_labels
        self.cnn_channels = cnn_channels
        self.fc_sizes = fc_sizes


class LeNet5PreTrainedModel(PreTrainedModel):
    config_class = LeNet5Config
    base_model_prefix = "lenet5"


class LeNet5Model(LeNet5PreTrainedModel):
    def __init__(self, config: LeNet5Config) -> None:
        super().__init__(config)
        self.conv1 = nn.Conv2d(
            config.in_channels, config.cnn_channels[0], kernel_size=5
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            config.cnn_channels[0], config.cnn_channels[1], kernel_size=5
        )
        self.fc1 = nn.Linear(config.fc_sizes[0], config.fc_sizes[1])
        self.fc2 = nn.Linear(config.fc_sizes[1], config.fc_sizes[2])

        self.post_init()

    def forward(
        self, pixel_values, labels=None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        x = torch.relu(self.conv1(pixel_values))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return BaseModelOutputWithPoolingAndNoAttention(
            pooler_output=x,
            last_hidden_state=None,
        )


class LeNet5ForImageClassification(LeNet5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.lenet = LeNet5Model(config)

        self.num_labels = config.num_labels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            (
                nn.Linear(config.fc_sizes[-1], config.num_labels)
                if config.num_labels > 0
                else nn.Idntity()
            ),
        )

        self.post_init()

    def forward(
        self, pixel_values: torch.FloatTensor, labels: torch.LongTensor | None = None
    ):
        outputs = self.lenet(pixel_values)

        pooler_output = outputs.pooler_output

        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits)


