"""Implementation of original U-Net paper."""
import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

#class to perform double convolution 
#Note: should this be its own class???
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        #nn.Sequential object to perform double convolution
        self.convolve = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1), #2D convolution
            nn.ReLU(inplace=True), #activation function
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1), #2D convolution
            nn.ReLU(inplace=True) #activation function
        )

    def forward(self, x):
        return self.convolve(x) #returning double convolution w/ activation functions

#OG U-Net class that inherits from nn.Module
class UNet(nn.Module):
    #class constructor
    def __init__(self, in_channels=3, out_channels=1, feature_sizes=[64, 128, 256, 512, 1024]):
        #note: kernel_size = filter_size
        super(UNet, self).__init__()
        
        #setting member variables
        self.contract = nn.ModuleList() #stores contracting network layers (downsampling)
        self.expand = nn.ModuleList() #stores expanding network layers (upsampling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #pooling layer used by U-Net 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_sizes = feature_sizes

        #setting double convs for contracting path
        for feature_size in feature_sizes:
            self.contract.append(DoubleConv(in_channels=self.in_channels, out_channels=feature_size))
            self.in_channels = feature_size

        #setting transposed convs (to upsample) and double convs for expanding path 
        for feature_size in reversed(feature_sizes):
            self.expand.append(
                nn.ConvTranspose2d(in_channels=feature_size, out_channels=(feature_size // 2), kernel_size=2, stride=2) #transposed conv
            )#append
            self.expand.append(DoubleConv(in_channels=feature_size, out_channels=(feature_size // 2))) #double conv

            #if statement to ensure correct dimensions given feature size array
            if len(self.expand) == 8:
                break
        
        self.final_layer = nn.Conv2d(in_channels=feature_sizes[0], out_channels=self.out_channels, kernel_size=1, stride=1) #final layer of U-Net

    #function to perform forward pass of UNet (Note: forward fcn. is inherited from nn.Module) 
    def forward(self, x): 
        skip_connections = [] #list to store skip connections

        #contracting path (downsample)
        for downsample in self.contract:
            x = downsample(x) #applying double conv and ReLU
            
            #storing skip connections and applying max pooling
            if len(skip_connections) < 4:
                skip_connections.append(x) #storing output for skip connections
                x = self.pool(x) #applying max pooling

        #reversing skip connections to index easier
        skip_connections = list(reversed(skip_connections))

        #expansive path (upsample)
        for i in range(len(self.expand)):
            x = self.expand[i](x)
            if (i % 2) == 0:
                skip = skip_connections[(i // 2)] #cropped skip connection
                if (x.shape != skip.shape): #checking if dimensions match
                    skip = CenterCrop(size=(x.shape[-2], x.shape[-1]))(skip) #center cropping skip connection to concat.

                x = torch.cat(tensors=(skip, x), dim=1) #concatenating skip connection to up-conv
                
        #final layer
        x = self.final_layer(x)

        return x

if __name__ == "__main__":
    #sanity check
    img_size = 572
    num_samples = 10
    num_channels = 3
    x = torch.randn((num_samples, num_channels, img_size, img_size)) #dummy variable to represent RGB images
    print(x.shape)

    out_channels = 1 #for binary mask
    model = UNet(in_channels=3, out_channels=out_channels) #initializing a UNet object
    preds = model(x)
    print(preds.shape)
    # print(model)
