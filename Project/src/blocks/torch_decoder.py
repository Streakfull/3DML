import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.LeakyReLU(0.01)

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
        x = nn.LeakyReLU(0.01)(x)
        return x

class TripleResidualBlock(nn.Module):
    def __init__(self, inp,kernel_size):
        super(TripleResidualBlock, self).__init__()
        inp_dim = inp
        
        self.conv1 = ConvBlock(inp, inp, kernel_size)
        self.conv2 = ConvBlock(inp, inp, kernel_size)
        self.conv3 = ConvBlock(inp, inp, kernel_size)
        #1x1 conv.
        self.convi = ConvBlock(inp, inp_dim, kernel_size=1)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
    
        x = x + shortcut
        x = self.convi(x)
        x = nn.LeakyReLU(0.1)(x)
        return x

# Define the model
class Decoder(nn.Module):
    n_deconvfilter = [128, 128, 128, 64, 32, 2]
    
    def __init__(self, in_channels=256, n_deconvfilter=n_deconvfilter):
        super(Decoder, self).__init__()
        
        #does not change the dimensions of tensor
        self.initialpool = nn.MaxPool3d(kernel_size=1, stride=1, padding=0, return_indices=True)

        self.conv1a = ConvBlock(in_channels,n_deconvfilter[1], kernel_size=3)
        self.conv1b = ConvBlock(n_deconvfilter[1], n_deconvfilter[1], kernel_size=3)
        self.res1 = ResidualBlock(n_deconvfilter[1], kernel_size=3)
        self.unpool1 = nn.MaxUnpool3d(kernel_size=2, padding=1)
        
        self.conv2a = ConvBlock(n_deconvfilter[1], n_deconvfilter[2], kernel_size=3)
        self.conv2b = ConvBlock(n_deconvfilter[2], n_deconvfilter[2], kernel_size=3)
        self.res2 = ResidualBlock(n_deconvfilter[2], kernel_size=3)
        self.unpool2 = nn.MaxUnpool3d(kernel_size=2, padding=1)

        self.conv3a = ConvBlock(n_deconvfilter[2], n_deconvfilter[3], kernel_size=3)
        self.conv3b = ConvBlock(n_deconvfilter[3], n_deconvfilter[3], kernel_size=3)
        self.res3 = ResidualBlock(n_deconvfilter[3], kernel_size=3)
        self.unpool3 = nn.MaxUnpool3d(kernel_size=2, padding=1)
        
        self.conv4a = ConvBlock(n_deconvfilter[3], n_deconvfilter[4], kernel_size=3)
        self.conv4b = ConvBlock(n_deconvfilter[4], n_deconvfilter[4], kernel_size=3)
        self.conv4c = ConvBlock(n_deconvfilter[4], n_deconvfilter[4], kernel_size=3)
        self.res4 = TripleResidualBlock(n_deconvfilter[4], kernel_size=3)
        self.unpool4 = nn.MaxUnpool3d(kernel_size=2, padding=2)
        
        self.conv5a = ConvBlock(n_deconvfilter[4], n_deconvfilter[5], kernel_size=3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        pooled_tensor, indices = self.initialpool(x)
        x = self.unpool1(pooled_tensor, indices)
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.res1(x)
   
   
        pooled_tensor, indices = self.initialpool(x)
        x = self.unpool2(pooled_tensor, indices)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.res2(x)

  
        pooled_tensor, indices = self.initialpool(x)
        x = self.unpool3(pooled_tensor, indices)
        x = self.conv3a(x)
        x = self.conv3b(x)

        x = self.res3(x)

        pooled_tensor, indices = self.initialpool(x)
        x = self.unpool4(pooled_tensor, indices)
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.conv4c(x)
        x = self.res4(x)
   
        #print("Output of layer4:", x.shape)
        
        x = self.conv5a(x)
        #print("Output of layer5:", x.shape)
        
        x = self.softmax(x)
        return x[:, 0, :, :, :]

# n_deconvfilter = [128, 128, 128, 64, 32, 2]
# # Create an instance of the model
# model = Decoder(3, n_deconvfilter=n_deconvfilter)
# x = torch.randn([1, 3, 4, 4, 4])
# output = model(x)

# # Print the model summary
# print(output.shape)