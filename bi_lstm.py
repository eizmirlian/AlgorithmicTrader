import torch
import torch.nn as nn

class bi_lstm(nn.Module):

    def __init__(self, input_size, batch_size, hidden_size, output_size, num_layers= 3):
        super(bi_lstm, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.init_linear = nn.Linear(input_size, input_size)

        self.lstm_layers = nn.LSTM(input_size, hidden_size, num_layers, bidirectional= True, dropout= .15)

        self.tanh_act = nn.Tanh()

        self.linear = nn.Linear(hidden_size * 2, output_size * 2)
        self.linear_out = nn.Linear(output_size * 2, output_size)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))


    def forward(self, input):
        init_out = self.init_linear(input)

        lstm_out, self.hidden = self.lstm_layers(init_out)

        act_out = self.tanh_act(lstm_out)

        lin_out = self.linear(act_out)

        y_pred = self.linear_out(lin_out)

        return y_pred
