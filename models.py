from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=(3 // 2), bias=bias)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=(3 // 2), bias=bias)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + x

class SRCNNResBlock(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNNResBlock, self).__init__()

        self.resblock1 = ResidualBlock(num_channels, 32)
        self.resblock2 = ResidualBlock(32, 32)
        self.resblock3 = ResidualBlock(32, num_channels)

    def forward(self, x):
        out = self.resblock1(x)
        out = self.resblock2(x)
        out = self.resblock3(x)
        out = out + x