import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A single computational LSTM core for the GQN generator architecture.
"""
class LSTMCore(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=5, stride=1, padding=2):
        super(LSTMCore, self).__init__()
        self.dim_in = dim_in + dim_out
        self.dim_out = dim_out

        self.forget = nn.Conv2d(self.dim_in, dim_out, kernel_size, padding=padding, stride=stride)
        self.input = nn.Conv2d(self.dim_in, dim_out, kernel_size, padding=padding, stride=stride)
        self.output = nn.Conv2d(self.dim_in, dim_out, kernel_size, padding=padding, stride=stride)
        self.state = nn.Conv2d(self.dim_in, dim_out, kernel_size, padding=padding, stride=stride)

    def forward(self, x, h, c):
        hx = torch.cat((h, x), dim=1)
        forget_gate = torch.sigmoid(self.forget(hx))
        input_gate = torch.sigmoid(self.input(hx))
        output_gate = torch.sigmoid(self.output(hx))
        state_gate = torch.tanh(self.state(hx))

        c = forget_gate * c + input_gate * state_gate
        h = output_gate * torch.tanh(c)
        return h, c

"""
A single GRU cell
"""
class GRUCore(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=5, stride=1, padding=2):
        super(GRUCore, self).__init__()
        self.dim_in = dim_in + dim_out
        self.dim_out = dim_out

        self.reset = nn.Conv2d(self.dim_in, dim_out, kernel_size, padding=padding, stride=stride)
        self.update = nn.Conv2d(self.dim_in, dim_out, kernel_size, padding=padding, stride=stride)
        self.state = nn.Conv2d(self.dim_in, dim_out, kernel_size, padding=padding, stride=stride)

    def forward(self, x, h):
        hx = torch.cat((h, x), dim=1)
        reset_gate = torch.sigmoid(self.reset(hx))
        update_gate = torch.sigmoid(self.update(hx))

        hr = h * reset_gate

        state_gate = torch.tanh(self.state(torch.cat((hr, x), dim=1)))
        h = h * (1 - update_gate) + state_gate * update_gate
        return h
