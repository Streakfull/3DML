import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, inp,kernel_size):
        super(ResidualBlock, self).__init__()
        inp_dim = inp
        
        self.conv1 = ConvBlock(inp, inp, kernel_size)
        self.conv2 = ConvBlock(inp, inp, kernel_size)
        #1x1 conv.
        self.conv3 = ConvBlock(inp, inp_dim, kernel_size=1)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
    
        x = x + shortcut
        x = self.conv3(x)
        x = nn.ReLU()(x)
        return x

# Define the model
class Encoder(nn.Module):
    n_convfilter = [96, 128, 256, 256, 256, 256]
    n_fc_filters = [1024]
    inp_channels = 3
    
    def __init__(self, in_channels, n_convfilter, n_fc_filters):
        super(Encoder, self).__init__()

        self.conv1a = ConvBlock(in_channels,n_convfilter[0], kernel_size=7)
        self.conv1b = ConvBlock(n_convfilter[0], n_convfilter[0], kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), padding=1)
        
        self.conv2a = ConvBlock(n_convfilter[0], n_convfilter[1], kernel_size=3)
        self.conv2b = ConvBlock(n_convfilter[1], n_convfilter[1], kernel_size=3)
        #1x1 conv. filter
        # self.conv2c = ConvBlock(n_convfilter[1], n_convfilter[1], kernel_size=1)
        self.res2 = ResidualBlock(n_convfilter[1], kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), padding=1)
        
        self.conv3a = ConvBlock(n_convfilter[1], n_convfilter[2], kernel_size=3)
        self.conv3b = ConvBlock(n_convfilter[2], n_convfilter[2], kernel_size=3)
        self.res3 = ResidualBlock(n_convfilter[2], kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), padding=1)
        
        self.conv4a = ConvBlock(n_convfilter[2], n_convfilter[3], kernel_size=3)
        self.conv4b = ConvBlock(n_convfilter[3], n_convfilter[3], kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), padding=1)
        
        self.conv5a = ConvBlock(n_convfilter[3], n_convfilter[4], kernel_size=3)
        self.conv5b = ConvBlock(n_convfilter[4], n_convfilter[4], kernel_size=3)
        self.res5 = ResidualBlock(n_convfilter[4], kernel_size=3)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,2), padding=1)
        
        self.conv6a = ConvBlock(n_convfilter[4], n_convfilter[5], kernel_size=3)
        self.conv6b = ConvBlock(n_convfilter[5], n_convfilter[5], kernel_size=3)
        self.res6 = ResidualBlock(n_convfilter[5], kernel_size=3)
        self.pool6 = nn.MaxPool2d(kernel_size=(2,2), padding=0, stride=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4096, 1024)
        
    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)
        print("Output of layer1:", x.shape)
        
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.res2(x)
        x = self.pool2(x)
        print("Output of layer2:", x.shape)
        
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.res3(x)
        x = self.pool3(x)
        print("Output of layer3:", x.shape)
        
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)
        print("Output of layer4:", x.shape)
        
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.res5(x)
        x = self.pool5(x)
        print("Output of layer5:", x.shape)
        
        x = self.conv6a(x)
        x = self.conv6b(x)
        x = self.res6(x)
        x = self.pool6(x)
        print("Output of layer6:", x.shape)
        
        x = self.flatten(x)
        x = self.fc(x)
        
        return x

# n_convfilter = [96, 128, 256, 256, 256, 256]
# n_fc_filters = [1024]
# # Create an instance of the model
# model = Encoder(3, n_convfilter=n_convfilter, n_fc_filters=n_fc_filters)
# x = torch.randn(1, 3, 127, 127)
# output = model(x)

# # Print the model summary
# print(output.shape)