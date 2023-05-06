import pytest
import torch

from src.model import BidirectionalLSTM, ConvBNReluBlock, ConvReluBlock


@pytest.fixture
def lstm_input():
    batch_size = 4
    seq_length = 32
    input_size = 256
    return torch.randn(batch_size, seq_length, input_size)


@pytest.fixture
def lstm_model():
    input_size = 256
    hidden_size = 128
    out_features = 64
    return BidirectionalLSTM(
        input_size=input_size, hidden_size=hidden_size, out_features=out_features
    )


def test_BidirectionalLSTM(lstm_input, lstm_model):
    output_tensor = lstm_model(lstm_input)
    assert output_tensor.shape == (
        lstm_input.shape[0],
        lstm_input.shape[1],
        lstm_model.out_features,
    )


@pytest.fixture()
def conv_relu_block():
    return ConvReluBlock(3, 16, kernel=(3, 3), stride=(1, 1), padding=(1, 1))


def test_forward(conv_relu_block):
    input = torch.randn(1, 3, 32, 32)
    output = conv_relu_block(input)
    assert output.shape == (1, 16, 32, 32)
    assert torch.min(output) >= 0


@pytest.fixture
def conv_bn_relu_block():
    return ConvBNReluBlock(
        in_channels=3,
        out_channels=64,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    )


def test_ConvBNReluBlock(conv_bn_relu_block):
    x = torch.randn((2, 3, 32, 32))
    output = conv_bn_relu_block(x)
    assert output.shape == (2, 64, 32, 32)
    assert torch.min(output) >= 0
