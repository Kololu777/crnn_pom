import torch.nn as nn
from torch import Tensor


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
        self.out_features = out_features
        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True
        )
        self.embedding = nn.Linear(in_features=hidden_size * 2, out_features=self.out_features)

    def forward(self, input: Tensor) -> Tensor:
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.embedding(recurrent)
        return output
