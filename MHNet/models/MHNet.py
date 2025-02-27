import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.module import MSCA, SISM, HBGM, DepthwiseSeparableConv
from modules.smt import smt_t



class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBNReLU, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Net(nn.Module):
    def __init__(self, dct_extractor):
        super(Net, self).__init__()
        self.smt = smt_t()
        self.dct_extractor = dct_extractor
        stage_channels = [64, 128, 256, 512]
        self.channel = 64
        self.shrink1 = nn.Conv2d(stage_channels[0], self.channel, kernel_size=1, stride=1)
        self.shrink2 = nn.Conv2d(stage_channels[1], self.channel, kernel_size=1, stride=1)
        self.shrink3 = nn.Conv2d(stage_channels[2], self.channel, kernel_size=1, stride=1)
        self.shrink4 = nn.Conv2d(stage_channels[3], self.channel, kernel_size=1, stride=1)
        self.spca1 = MSCA(self.channel)
        self.spca2 = MSCA(self.channel)
        self.spca3 = MSCA(self.channel)
        self.spca4 = MSCA(self.channel)
        self.loc43 = SISM(self.channel * 2, self.channel)
        self.conv3_1 = ConvBNReLU(self.channel * 2, self.channel)
        self.conv3_2 = ConvBNReLU(self.channel * 2, self.channel)
        self.conv3_3 = ConvBNReLU(self.channel * 2, self.channel)
        self.conv3_4 = ConvBNReLU(self.channel * 2, self.channel)
        self.freqconv = nn.Sequential(
            DepthwiseSeparableConv(96, 96, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(96, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.egb1 = HBGM(in_channels=self.channel)
        self.egb2 = HBGM(in_channels=self.channel)
        self.egb3 = HBGM(in_channels=self.channel)
        self.decoder1 = ConvBNReLU(self.channel * 2, self.channel)
        self.decoder2 = ConvBNReLU(self.channel * 2, self.channel)
        self.decoder3 = ConvBNReLU(self.channel * 2, self.channel)
        self.decoder4 = ConvBNReLU(self.channel, self.channel)
        self.reconv1 = ConvBNReLU(self.channel * 2, self.channel)
        self.reconv2 = ConvBNReLU(self.channel * 2, self.channel)
        self.reconv3 = ConvBNReLU(self.channel * 2, self.channel)
        self.out_conv1 = nn.Conv2d(self.channel, 1, kernel_size=1)
        self.out_conv2 = nn.Conv2d(self.channel, 1, kernel_size=1)
        self.out_conv3 = nn.Conv2d(self.channel, 1, kernel_size=1)
        self.out_conv4 = nn.Conv2d(self.channel, 1, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        high, low = self.dct_extractor(x)
        freq = self.freqconv(high)
        x1, x2, x3, x4 = self.smt(x)
        x1 = self.shrink1(x1)
        x2 = self.shrink2(x2)
        x3 = self.shrink3(x3)
        x4 = self.shrink4(x4)
        xl = self.loc43(x3, self.up(x4))
        s4 = self.spca4(
            self.conv3_4(torch.cat((F.interpolate(xl, size=(12, 12), mode='bilinear', align_corners=True), x4), 1)))
        s3 = self.spca3(self.conv3_3(torch.cat((self.up(s4), x3), 1)))
        s2 = self.spca2(self.conv3_2(torch.cat((self.up(s3), x2), 1)))
        s1 = self.spca1(self.conv3_1(torch.cat((self.up(s2), x1), 1)))
        d4 = self.decoder4(s4)
        d4_up = self.up(d4)
        out4 = self.out_conv4(d4)
        pmask4 = F.interpolate(out4, size=(384, 384), mode='bilinear', align_corners=True)
        p4 = pmask4
        egb3, edge3 = self.egb3(s3, p4, freq)
        edge3 = F.interpolate(edge3, size=(384, 384), mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat((d4_up, egb3), dim=1))
        rd3 = self.reconv3(torch.cat((d4_up, d3), 1))
        out3 = self.out_conv3(rd3)
        pmask3 = F.interpolate(out3, size=(384, 384), mode='bilinear', align_corners=True)
        p3 = pmask3 + p4
        egb2, edge2 = self.egb2(s2, p3, freq)
        edge2 = F.interpolate(edge2, size=(384, 384), mode='bilinear', align_corners=True)
        d3_up = self.up(rd3)
        d2 = self.decoder2(torch.cat((d3_up, egb2), dim=1))
        rd2 = self.reconv2(torch.cat((d3_up, d2), 1))
        out2 = self.out_conv2(rd2)
        pmask2 = F.interpolate(out2, size=(384, 384), mode='bilinear', align_corners=True)
        p2 = pmask2 + p3
        egb1, edge1 = self.egb1(s1, p2, freq)
        edge1 = F.interpolate(edge1, size=(384, 384), mode='bilinear', align_corners=True)
        d2_up = self.up(rd2)
        d1 = self.decoder1(torch.cat((d2_up, egb1), dim=1))
        rd1 = self.reconv1(torch.cat((d2_up, d1), 1))
        out1 = self.out_conv1(rd1)
        pmask1 = F.interpolate(out1, size=(384, 384), mode='bilinear', align_corners=True)
        p1 = pmask1 + p2

        return p1, p2, p3, p4, edge1, edge2, edge3

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)['model'])
        print(f"loading pre_model ${pre_model}")


if __name__ == '__main__':
    from thop import profile
    import pickle
    from modules.module import DCTFeatureExtractor

    with open('../freq_mean_std.pkl', 'rb') as f:
        freq_stats = pickle.load(f)
        freq_mean = torch.tensor(freq_stats['mean'])
        freq_std = torch.tensor(freq_stats['std'])
    dct_module = DCTFeatureExtractor(freq_mean, freq_std)
    model = Net(dct_module)
    inputs = torch.randn(size=(1, 3, 384, 384))
    flops, params = profile(model, (inputs,))
    print('flops: %.3f G, parms: %.3f M' % (flops / 1000000000.0, params / 1000000.0))
    # flops: 12.798 G, parms: 12.474 M
