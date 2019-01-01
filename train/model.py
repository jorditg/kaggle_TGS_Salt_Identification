"""
With inspiration from https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/albu/src/pytorch_zoo/linknet.py
"""

import importlib
import torch.nn as nn

nonlinearity = nn.ReLU

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class LinkNet101x101(nn.Module):
    def __init__(self, num_classes, num_channels=1, encoder='resnet18_lite_101x101',
                 final='sigmoid'):
        super().__init__()
        assert num_channels > 0, "Incorrect num channels"
        assert encoder in ['resnet18_lite_101x101', 'resnet34_lite_101x101'],\
                           "Incorrect encoder type"
        assert final in ['softmax', 'sigmoid'],\
                         "Incorrect output type"

        if encoder in ['resnet18_lite_101x101', 'resnet34_lite_101x101']:
            filters = [64, 64, 64, 64]
            resnet = class_for_name("resnet_101x101", encoder)()
            
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
#        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3,
                                               stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.ConvTranspose2d(32, 32, 4)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        if final=='softmax': # softmax for multiclass classification
            self.final = nn.Softmax(dim=1)
        else: # one class classifier
            self.final = nn.Sigmoid()

    def forward(self, x):
        #x = self.pad(x)
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = d2#self.decoder1(d2)

        # Final Classification
#        print(d1.size())
        x = self.finaldeconv1(d1)
        x = self.finalrelu1(x)
#        print(x.size())
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
#        print(x.size())        
        x = self.finalconv3(x)
#        print(x.size())        
        return self.final(x)
