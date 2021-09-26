import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class Linear(nn.Module):
    def __init__(self, name, args=True):
        super(Linear, self).__init__()
        self.name = name
        [self.n_dim, self.n_out] = args
        self.fc = nn.Linear(self.n_dim, self.n_out)

    def forward(self, x):
        out = self.fc(x)
        return out


class MnistNet(nn.Module):
    def __init__(self, name, args=True):
        super(MnistNet, self).__init__()
        self.name = name
        self.n_cls = 10
        self.fc1 = nn.Linear(1 * 28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.n_cls)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        fc1_out = F.relu(self.fc1(x))
        fc2_out = F.relu(self.fc2(fc1_out))
        out = self.fc3(fc2_out)
        return out


class EmnistNet(nn.Module):
    def __init__(self, name, args=True):
        super(EmnistNet, self).__init__()
        self.name = name
        self.n_cls = 10
        self.fc1 = nn.Linear(1 * 28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, self.n_cls)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        fc1_out = F.relu(self.fc1(x))
        fc2_out = F.relu(self.fc2(fc1_out))
        out = self.fc3(fc2_out)
        return out


class Cifar10Net(nn.Module):
    def __init__(self, name, args=True):
        super(Cifar10Net, self).__init__()
        self.name = name
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        conv1_out = self.pool(F.relu(self.conv1(x)))
        conv2_out = self.pool(F.relu(self.conv2(conv1_out)))
        conv2_out = conv2_out.view(-1, 64 * 5 * 5)
        fc1_out = F.relu(self.fc1(conv2_out))
        fc2_out = F.relu(self.fc2(fc1_out))
        out = self.fc3(fc2_out)
        return out


class Cifar100Net(nn.Module):
    def __init__(self, name, args=True):
        super(Cifar100Net, self).__init__()
        self.name = name
        self.n_cls = 100
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        conv1_out = self.pool(F.relu(self.conv1(x)))
        conv2_out = self.pool(F.relu(self.conv2(conv1_out)))
        conv2_out = conv2_out.view(-1, 64 * 5 * 5)
        fc1_out = F.relu(self.fc1(conv2_out))
        fc2_out = F.relu(self.fc2(fc1_out))
        out = self.fc3(fc2_out)
        return out


class ResNet18(nn.Module):
    def __init__(self, name, args=True):
        super(ResNet18, self).__init__()
        self.name = name
        resnet18 = models.resnet18()
        resnet18.fc = nn.Linear(512, 10)

        # Change BN to GN
        resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

        assert len(dict(resnet18.named_parameters()).keys()) == len(
            resnet18.state_dict().keys()), 'More BN layers are there...'
        self.model = resnet18

    def forward(self, x):
        out = self.model(x)
        return out


class ShakeSpeareNet(nn.Module):
    def __init__(self, name, args=True):
        super(ShakeSpeareNet, self).__init__()
        self.name = name
        embedding_dim = 8
        hidden_size = 100
        num_LSTM = 2
        input_length = 80
        self.n_cls = 80

        self.embedding = nn.Embedding(input_length, embedding_dim)
        self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                                    num_layers=num_LSTM)
        self.fc = nn.Linear(hidden_size, self.n_cls)

    def forward(self, x):
        emb_out = self.embedding(x)
        emb_out = emb_out.permute(1, 0, 2)  # lstm accepts in this style
        output, (h_, c_) = self.stacked_LSTM(emb_out)
        # Choose last hidden layer
        last_hidden = output[-1, :, :]
        out = self.fc(last_hidden)
        return out
