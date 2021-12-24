from torch import nn
from torch.nn import init
import torch

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).contiguous()

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

class GAMAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, rate):
        super(GAMAllLayer, self).__init__()

        self.num = len(in_channels)
        if self.num == 1:
            self.gatt = GAM_Attention(in_channels[0], out_channels[0], rate)
        else:
            for i in range(self.num):
                self.add_module('gatt'+str(i+2),
                                GAM_Attention(in_channels[i],out_channels[i], rate))
    def forward(self, x):
        if self.num == 1:
            return self.gatt(x)
        else:
            out = []
            for i in range(self.num):
                gatt_layer = getattr(self, 'gatt'+str(i+2))
                out.append(gatt_layer(x[i]).contiguous())
            return out


if __name__ == '__main__':
    x = torch.randn(1, 64, 32, 48)
    b, c, h, w = x.shape
    net = GAM_Attention(in_channels=c, out_channels=c)
    y = net(x)
    print(y.size())