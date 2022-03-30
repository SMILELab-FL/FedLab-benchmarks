from torch import nn

# Model's architecture refers to https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/simulation/models/mnist.py
class EmnistCNN(nn.Module):
    def __init__(self):
        super(EmnistCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(in_features=1024, out_features=62),
        )

    def forward(self, x):
        return self.net(x)
