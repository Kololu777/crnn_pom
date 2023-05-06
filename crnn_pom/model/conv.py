from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t


class ConvReluBlock(nn.Module):
    """
    A block that performs convolution followed by ReLU, with the structure: nn.Conv2d -> nn.ReLU.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel (_size_2_t): The kernel size.
        stride (_size_2_t): The stride.
        padding (Union[_size_2_t, str]): The padding.
    Attributes:
        conv (nn.Conv2d): A convolutional layer.
        relu (nn.ReLU): A ReLU activation function.

    Examples:
        >>> block = ConvReluBlock(in_channels=3, out_channels=32, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        >>> input_tensor = torch.randn((1, 3, 32, 32))
        >>> output_tensor = block(input_tensor)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: _size_2_t,
        stride: _size_2_t,
        padding: Union[_size_2_t, str],
    ):
        super(ConvReluBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvBNReluBlock(nn.Module):
    """
    A block that performs convolution followed by Batch Normalization and ReLU,
    with the structure: nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel (_size_2_t): The kernel size.
        stride (_size_2_t): The stride.
        padding (Union[_size_2_t, str]): The padding.

    Attributes:
        conv (nn.Conv2d): Convolutional layer module.
        bn (nn.BatchNorm2d): Batch normalization layer module.
        relu (nn.ReLU): ReLU activation function module.

    Examples:
        >>> block = ConvBNReluBlock(in_channels=3, out_channels=16, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        >>> input_tensor = torch.randn((1, 3, 32, 32))
        >>> output = block(input_tensor)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: _size_2_t,
        stride: _size_2_t,
        padding: Union[_size_2_t, str],
    ):
        super(ConvBNReluBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BidirectionalLSTM(nn.Module):
    """
    A bidirectional LSTM module.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        out_features (int): The size of the output features.

    Attributes:
        rnn (nn.LSTM): LSTM layer.
        embedding (nn.Linear): Linear layer to generate output.

    Example:
        >>> lstm = BidirectionalLSTM(input_size=10, hidden_size=20, out_features=5)
        >>> x = torch.randn(2, 5, 10)  # batch_size x sequence_length x input_size
        >>> out = lstm(x)  # batch_size x sequence_length x out_features
    """

    def __init__(self, input_size: int, hidden_size: int, out_features: int):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.embedding = nn.Linear(
            in_features=hidden_size * 2, out_features=out_features
        )

    def forward(self, input: Tensor) -> Tensor:
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.embedding(recurrent)
        return output
