from torch import nn


class LSTMWrapper(nn.Module):
    

    def __init__(self, lstm_module: nn.LSTM):
        super().__init__()
        self.lstm = lstm_module

    def forward(self, input, h_0=None, c_0=None):
        (seq_len, batch, input_size) = input.shape
        num_layers = 1
        num_directions = self.lstm.bidirectional + 1
        hidden_size = self.lstm.hidden_size
        if h_0 is None or c_0 is None or h_0.numel() == 0 or c_0.numel() == 0:
            tuple_0 = None
        else:
            tuple_0 = (h_0, c_0)
        output, (h_n, c_n) = self.lstm(input, tuple_0)

        Y = output.view(seq_len, batch, num_directions, hidden_size).transpose(1, 2)
        Y_h = h_n.view(num_layers, num_directions, batch, hidden_size).squeeze(0)
        Y_c = c_n.view(num_layers, num_directions, batch, hidden_size).squeeze(0)
        return Y, Y_h, Y_c
