import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import resnet
from flgc import Flgc2d
from dwt import DWT
from fam import FAM_Module


__all__ = ["LinkNet"]


class LPDGC(nn.Module):
    def __init__(self, inplanes, planes):
        super(LPDGC, self).__init__()
        self.dilated_conv_1 = Flgc2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = Flgc2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilated_conv_3 = Flgc2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_4 = Flgc2d(inplanes, planes, kernel_size=3, stride=1, padding=4, dilation=4)
        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        self.relu3 = nn.ELU(inplace=True)
        self.relu4 = nn.ELU(inplace=True)

    def forward(self, x):
        out1 = self.dilated_conv_1(x)
        out2 = self.dilated_conv_2(x)
        out3 = self.dilated_conv_3(x)
        out4 = self.dilated_conv_4(x)
        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        out3 = self.relu3(out3)
        out4 = self.relu4(out4)
        out = out1 + out2 + out3 + out4
        return out

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out+residual)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True))
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True))

    def forward(self, x_high_level, x_low_level):
        x = self.conv1(x_high_level)
        x = self.tp_conv(x)
        x = center_crop(x, x_low_level.size()[2], x_low_level.size()[3])
        x = self.conv2(x)

        return x

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    diffy = (h - max_height) // 2
    diffx = (w -max_width) // 2
    return layer[:,:,diffy:(diffy + max_height),diffx:(diffx + max_width)]


def up_pad(layer, skip_height, skip_width):
    _, _, h, w = layer.size()
    diffy = skip_height - h
    diffx = skip_width -w
    return F.pad(layer,[diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])


class LungINFseg(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, classes=19):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super().__init__()
        self.dwt = DWT()
        self.reshape= nn.Conv2d(12, 3, kernel_size=3,padding=1, bias=False)

        base = resnet.resnet18(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.sz1 = nn.Conv2d(3, 16, kernel_size=3,padding=1, bias=False)
        self.sz2 = nn.Conv2d(3, 32, kernel_size=3,padding=1, bias=False)
        self.sz3 = nn.Conv2d(3, 64, kernel_size=3,padding=1, bias=False)
        self.sz4 = nn.Conv2d(3, 128, kernel_size=3,padding=1, bias=False)


        self.encoder1 = base.layer1    
        self.dilated2_1 = LPDGC(64, 64)
        self.cam1 = FAM_Module(64)

        self.encoder2 = base.layer2  
        self.dilated2_2 = LPDGC(128, 128)
        self.cam2 = FAM_Module(128)

        self.encoder3 = base.layer3  
        self.dilated2_3 = LPDGC(256, 256)
        self.cam3 = FAM_Module(256)
        
        self.encoder4 = base.layer4 
        self.dilated2_4 = LPDGC(512, 512)
        self.cam4 = FAM_Module(512)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, classes, 2, 2, 0)


    def forward(self, x):
        input = x
        x = self.dwt(x)
        x = self.reshape(x)
        x = nn.functional.interpolate(x, size=(256,256), mode='bilinear', align_corners=True)
        x = self.in_block(x)
# ============================================================================================
        # RFA blocks 1
        e1 = self.encoder1(x)
        a = self.sz1(input)
        x1 = nn.functional.interpolate(a, size=64, mode='bilinear', align_corners=True)  
        e1_re1 = self.dwt(x1)
        e1_re1 = nn.functional.interpolate(e1_re1, size=64, mode='bilinear', align_corners=True)             
        e1 = e1_re1 + e1
        e1 = self.dilated2_1(e1)
        e1 = self.cam1(e1)
# ============================================================================================
        # RFA blocks 2
        e2 = self.encoder2(e1)
        b = self.sz2(input) 
        x2 = nn.functional.interpolate(b, size=32, mode='bilinear', align_corners=True)  
        e2_re2 = self.dwt(x2)
        e2_re2 = nn.functional.interpolate(e2_re2, size=32, mode='bilinear', align_corners=True) 
        e2 = e2_re2 + e2
        e2 = self.dilated2_2(e2)
        e2 = self.cam2(e2)
# ============================================================================================
        # RFA blocks 3
        e3 = self.encoder3(e2)
        c = self.sz3(input) 
        x3 = nn.functional.interpolate(c, size=16, mode='bilinear', align_corners=True)  
        e3_re3 = self.dwt(x3)
        e3_re3 = nn.functional.interpolate(e3_re3, size=16, mode='bilinear', align_corners=True) 
        e3 = e3_re3 + e3
        e3 = self.dilated2_3(e3)
        e3 = self.cam3(e3)
# ============================================================================================
        # RFA blocks 4
        e4 = self.encoder4(e3)
        d = self.sz4(input) 
        x4 = nn.functional.interpolate(d, size=8, mode='bilinear', align_corners=True)  
        e4_re4 = self.dwt(x4)
        e4_re4 = nn.functional.interpolate(e4_re4, size=8, mode='bilinear', align_corners=True) 
        e4 = e4_re4 + e4
        e4 = self.dilated2_4(e4)
        e4 = self.cam4(e4)
# ============================================================================================
        # Decoder blocks
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)
        return e1,e2,e3,e4, y
        
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LungINFseg(classes=1).to(device)
#     summary(model,(3,256,256))
