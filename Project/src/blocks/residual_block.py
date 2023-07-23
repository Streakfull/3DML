from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, kernel_size = 3, padding = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv3d(input_channels, input_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv3d(input_channels, input_channels, 1)
        )

    def forward(self, x):
        return self.net(x) + x