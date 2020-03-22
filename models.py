import torch

class Network(torch.nn.Module):
    def __init__(self, activation=None, input=1, layers=2, hidden=50, output=2):
        super(Network, self).__init__()
        if activation is None:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = activation

        self.fca = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            self.activation
        )

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input, hidden),
            *[self.fca for _ in range(layers)],
            torch.nn.Linear(hidden, output)
        )

    def forward(self, x):
        x = self.ffn(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x