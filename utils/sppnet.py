# https://github.com/danmohaha/DSP-FWA/blob/master/py_utils/DL/pytorch_utils/models/classifier.pyid
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import os, math

''''
folder='/Users/dhanley2/Documents/Personal/dfake/weights'
for bb in [18, 34, 50]:    
    mod = ResNet(bb, num_class=2, pretrained=True, folder=folder)
    output_model_file = '{}/resnet{}.pth'.format(folder, bb)
    torch.save(mod.state_dict(), output_model_file)
'''
class ResNet(nn.Module):
    def __init__(self, layers=18, num_class=2, pretrained=False, folder = None):
        super(ResNet, self).__init__()
        if layers == 18:
            self.resnet = models.resnet18(pretrained=pretrained)
        elif layers == 34:
            self.resnet = models.resnet34(pretrained=pretrained)
        elif layers == 50:
            self.resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            self.resnet = models.resnet101(pretrained=pretrained)
        elif layers == 152:
            self.resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError('layers should be 18, 34, 50, 101.')
        self.num_class = num_class
        if layers in [18, 34]:
            self.fc = nn.Linear(512, num_class)
        if layers in [50, 101, 152]:
            self.fc = nn.Linear(512 * 4, num_class)

    def conv_base(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)
        return layer1, layer2, layer3, layer4

    def forward(self, x):
        layer1, layer2, layer3, layer4 = self.conv_base(x)
        x = self.resnet.avgpool(layer4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class DensNet(nn.Module):
    # def __init__(self, ckpt, num_classes=2, pretrained=False, folder = None):
    def __init__(self, ckpt, layers=18, num_class=2, pretrained=False):
        super().__init__()
        if layers == 121:
            preloaded = models.densenet121(pretrained=pretrained)
        elif layers == 169:
            preloaded= models.densenet169(pretrained=pretrained)
        elif layers == 201:
            preloaded = models.densenet201(pretrained=pretrained)

        # preloaded = models.densenet121(pretrained)
        preloaded.load_state_dict(torch.load( ckpt ))

        self.features = preloaded.features
        # self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024, num_class, bias=True)
        del preloaded
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

# torch.load(os.path.join(folder, 'resnet{}.pth'.format(backbone)))

class SPPNet(nn.Module):
    def __init__(self, folder, architecture = 'resnet', backbone=50, num_class=2, \
                 pool_size=(1, 2, 6), pretrained=False):
        # Only resnet is supported in this version
        super(SPPNet, self).__init__()
        if architecture == 'resnet':
            if backbone in [18, 34, 50, 101, 152]:
                self.model = ResNet(backbone, num_class, pretrained, folder)
                self.model.load_state_dict(torch.load( os.path.join(folder, '{}{}.pth'.format(architecture, backbone))))
            else:
                raise ValueError('{}{} is not supported yet.'.format(architecture, backbone))

            backbones = {18:512, 34:512, 50:2048, 101:2048, 152:2048}
            self.c = backbones[backbone]
                
        elif  architecture == 'densenet':
            if backbone in [121, 169, 201]:
                ckpt = os.path.join(folder, '{}{}.pth'.format(architecture, backbone))
                self.model = DensNet(ckpt = ckpt, layers=backbone, num_class=num_class, pretrained=pretrained)
                # self.resnet.load_state_dict(torch.load( os.path.join(folder, '{}{}.pth'.format(architecture, backbone))))
            else:
                raise ValueError('{}{} is not supported yet.'.format(architecture, backbone))
                
            backbones = {121:1024, 169:1664, 201:1920}
            self.c = backbones[backbone]

        self.spp = SpatialPyramidPool2D(out_side=pool_size)
        #num_features = self.c * (pool_size[0] ** 2 + pool_size[1] ** 2 + pool_size[2] ** 2)
        #self.classifier = nn.Linear(num_features, num_class)

    def forward(self, x):
        if architecture == 'resnet':
            _, _, _, x = self.model.conv_base(x)
        elif architecture == 'densenet':
            features = self.features(x)
            x = F.relu(features, inplace=True)
        x = self.spp(x)
        # x = self.classifier(x)
        return x


class SpatialPyramidPool2D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        # batch_size, c, h, w = x.size()
        out = None
        for n in self.out_side:
            w_r, h_r = map(lambda s: math.ceil(s / n), x.size()[2:])  # Receptive Field Size
            s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out
