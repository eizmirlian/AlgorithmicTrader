import torch
import torch.nn as nn
import math

#Hyperparameters
LOOK_BACK = 80
BATCH_SIZE = 30
HIDDEN_SIZE = 30

class bi_lstm(nn.Module):

    def __init__(self, input_size= LOOK_BACK, batch_size= BATCH_SIZE, hidden_size= HIDDEN_SIZE, output_size= 1, num_layers= 3):
        super(bi_lstm, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.montecarlo = False

        self.init_linear = nn.Linear(input_size, input_size)

        self.lstm_layers = nn.LSTM(input_size, hidden_size, num_layers, bidirectional= True, batch_first= True, dropout= .1)

        #self.tanh_act = nn.Tanh()
        self.mc_dropout = nn.Dropout1d(.1)

        self.linear = nn.Linear(hidden_size * 2, output_size * 2)
        self.linear_out = nn.Linear(output_size * 2, output_size)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))


    def forward(self, input):
        #act_out = self.tanh_act(lstm_out)
        if self.montecarlo:
            init_out = self.init_linear(input)
            #init_out = self.mc_dropout(init_out)
            lstm_out, self.hidden = self.lstm_layers(init_out)
            lstm_out = self.mc_dropout(lstm_out)
            lin_out = self.linear(lstm_out)
            #lin_out = self.mc_dropout(lin_out)
        else:
            init_out = self.init_linear(input)
            lstm_out, self.hidden = self.lstm_layers(init_out)
            lin_out = self.linear(lstm_out)

        y_pred = self.linear_out(lin_out)

        return y_pred
    
    def dropout_on(self):
        self.mc_dropout.train()
        self.montecarlo = True
        self.lstm_layers.train()

    def dropout_off(self):
        self.montecarlo = False
        self.lstm_layers.eval()

class custom_loss(nn.Module):
#currently unused
    def __init__(self):
        super(custom_loss, self).__init__()

    def forward(self, output, target):
        loss = []
        mse = nn.MSELoss()
        for i in range(output.size(0)):
            for j in range(output[i].size(0)):
                if (output[i][j] < 0 and target[i][j] > 0) or (output[i][j] > 0 and target[i][j] < 0):
                    loss = math.pow(math.fabs(output[i][j]) + math.fabs(target[i][j]), 2)
                else:
                    loss += math.pow(math.fabs(output[i][j]) - math.fabs(target[i][j]), 2)
        return math.sqrt(loss)
