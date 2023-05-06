import torch.nn as nn
from einops import rearrange

from .recipe import (
    cnn_component_list_factory,
    factory_block,
    lstm_component_list_factory,
)

flavour = "pytorch"


class CRNN(nn.Module):
    """
    A Convolutional Recurrent Neural Network (CRNN) implementation using PyTorch.

    Args:
        cnn_recipe (dict): A dictionary specifying the recipe for the CNN.
        rnn_recipe (dict): A dictionary specifying the recipe for the RNN.

    Attributes:
        layers (nn.Sequential): A sequence of CNN layers.
        rnn (nn.Sequential): A sequence of RNN layers.

    Methods:
        forward(x): Implements the forward pass of the CRNN.

    Example:
        cnn_recipe = dict(
            conv1_x=[
                dict(type="ConvReLU", out_channels=64, kernel=3, padding=1, stride=1),
                dict(type="MaxPooling", kernel=(2, 2), stride=(2, 2), padding=(0, 0)),
            ],
            conv2_x=[
                dict(type="ConvReLU", out_channels=128, kernel=3, padding=1, stride=1),
                dict(type="MaxPooling", kernel=(2, 2), stride=(2, 2), padding=(0, 0)),
            ],
            ...
        )
        rnn_recipe = dict(
            lstm_x=[
                dict(type="BidirectionalLSTM", hidden_size=256, out_features=256),
                dict(type="BidirectionalLSTM", hidden_size=256, out_features=-1),
            ]
        )
        crnn = CRNN(cnn_recipe=cnn_recipe, rnn_recipe=rnn_recipe)
    """

    def __init__(self, cnn_recipe: dict, rnn_recipe: dict, input_channels, out_features):
        super(CRNN, self).__init__()
        layers = []
        rnn = []
        # Generate CNN layers
        cnn_recipe, out_channel = cnn_component_list_factory(cnn_recipe, input_channels=input_channels)
        for key, _ in cnn_recipe.items():
            for operator in cnn_recipe[key]:
                layers += [factory_block(**operator)]

        # Generate RNN layers
        rnn_recipe = lstm_component_list_factory(rnn_recipe, input_size=out_channel, out_features=out_features)
        for key, _ in rnn_recipe.items():
            for operator in rnn_recipe[key]:
                rnn += [factory_block(**operator)]
        self.layers = nn.Sequential(*layers)
        self.rnn = nn.Sequential(*rnn)

    def forward(self, x):
        # CNN Layers
        for layer in self.layers:
            x = layer(x)
        # Rearrange
        x = rearrange(x, "b c 1 w -> b c w")
        x = rearrange(x, "b c w -> b w c")
        # RNN Layers
        x = self.rnn(x)
        # Rearrange(CTC Loss format(T(w), N(b), C(c)))
        x = rearrange(x, "b w c -> w b c")
        return x
