import torch.nn as nn

from .conv import BidirectionalLSTM, ConvBNReluBlock, ConvReluBlock

crnn_cnn = dict(
    conv1_x=[
        dict(type="ConvReLU", out_channels=64, kernel=3, padding=1, stride=1),
        dict(type="MaxPooling", kernel=(2, 2), stride=(2, 2), padding=(0, 0)),
    ],
    conv2_x=[
        dict(type="ConvReLU", out_channels=128, kernel=3, padding=1, stride=1),
        dict(type="MaxPooling", kernel=(2, 2), stride=(2, 2), padding=(0, 0)),
    ],
    conv3_x=[
        dict(type="ConvReLU", out_channels=256, kernel=3, padding=1, stride=1),
        dict(type="ConvReLU", out_channels=256, kernel=3, padding=1, stride=1),
        dict(type="MaxPooling", kernel=(2, 1), stride=(2, 1), padding=(0, 0)),
    ],
    covn4_x=[
        dict(type="ConvBNReLU", out_channels=512, kernel=3, padding=1, stride=1),
        dict(type="ConvBNReLU", out_channels=512, kernel=3, padding=1, stride=1),
        dict(
            type="MaxPooling",
            out_channels=512,
            kernel=(2, 1),
            stride=(2, 1),
            padding=(0, 0),
        ),
    ],
    conv5_x=[dict(type="ConvBNReLU", out_channels=512, kernel=2, padding=0, stride=1)],
)

crnn_lstm = dict(
    lstm_x=[
        dict(type="BidirectionalLSTM", hidden_size=256, out_features=256),
        dict(type="BidirectionalLSTM", hidden_size=256, out_features=-1),
    ]
)


def cnn_component_list_factory(component_list: dict, input_channels: int) -> dict:
    """
    Constructs a list of convolutional components with input channel sizes and
    output channel sizes based on a recipe dictionary and input channel size.

    Args:
        component_list (dict): A dictionary containing a recipe for the
            convolutional component list.
        input_channels (int): The number of input channels.

    Returns:
        dict: A dictionary containing the list of convolutional components.
        last_out_channel: The number of output channels.
    """
    last_out_channel = input_channels
    for k, _ in component_list.items():
        for component in component_list[k]:
            component["in_channels"] = last_out_channel
            if "out_channels" not in component.keys():
                component["out_channels"] = last_out_channel
            else:
                last_out_channel = component["out_channels"]
    return component_list, last_out_channel


def lstm_component_list_factory(
    component_list: dict, input_size: int, out_features: int
) -> dict:
    """
    Constructs a list of LSTM components with input size, hidden size, and
    output feature size based on a recipe dictionary and input and output
    feature sizes.

    Args:
        component_list (dict): A dictionary containing a recipe for the LSTM
            component list.
        input_size (int): The number of input features.
        out_features (int): The number of output features.

    Returns:
        dict: A dictionary containing the list of LSTM components.
    """
    for k, _ in component_list.items():
        for component in component_list[k]:
            component["input_size"] = input_size
            input_size = component["out_features"]
        component_list[k][-1]["out_features"] = out_features
    return component_list


def factory_block(**kwargs):
    """
    Creates an instance of a specified block type with the given keyword arguments.

    Args:
        **kwargs: Keyword arguments containing the block type and its parameters.

    Returns:
        nn.Module: The instance of the specified block.

    Raises:
        ValueError: If the block type is not one of the supported types.
    """
    block_name = kwargs["type"]
    kwargs.pop("type")
    if block_name == "ConvReLU":
        return ConvReluBlock(**kwargs)
    elif block_name == "ConvBNReLU":
        return ConvBNReluBlock(**kwargs)
    elif block_name == "MaxPooling":
        kwargs["kernel_size"] = kwargs["kernel"]
        del kwargs["kernel"], kwargs["out_channels"], kwargs["in_channels"]
        return nn.MaxPool2d(**kwargs)
    elif block_name == "BidirectionalLSTM":
        return BidirectionalLSTM(**kwargs)
    else:
        raise ValueError(f"Unsupported block type: {block_name}")
