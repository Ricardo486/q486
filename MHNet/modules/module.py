import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.fft


class MSCA(nn.Module):
    def __init__(self, in_planes, ratio=16):  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        super(MSCA, self).__init__()
        # HPC
        self.avg_pool_3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool_5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.avg_pool_7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.section = in_planes // 4
        self.conv1 = nn.Conv2d(in_planes // 4, in_planes // 4, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes // 4, in_planes // 4, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_planes // 4, in_planes // 4, 3, padding=1, bias=False)
        self.conv7 = nn.Conv2d(in_planes // 4, in_planes // 4, 3, padding=1, bias=False)
        self.pool_h1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_h2 = nn.AdaptiveAvgPool2d((None, 2))
        self.pool_w1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w2 = nn.AdaptiveAvgPool2d((2, None))
        self.relu = nn.ReLU(inplace=True)
        self.fc1_h = nn.Conv2d(in_planes * 3, in_planes // ratio, kernel_size=1, padding=0)
        self.fc1_w = nn.Conv2d(in_planes * 3, in_planes // ratio, kernel_size=1, padding=0)
        self.fc2_h = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, padding=0)
        self.fc2_w = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.size()
        out_h1 = self.pool_h1(x)
        out_h2 = self.pool_h2(x)
        out_h2_split = out_h2.view(B, -1, H, 1)
        out_h = torch.cat((out_h1, out_h2_split), 1)
        out_h = self.fc1_h(out_h)
        out_h = self.relu(out_h)
        out_h = self.fc2_h(out_h)
        out_w1 = self.pool_w1(x)
        out_w2 = self.pool_w2(x)
        out_w2_split = out_w2.view(B, -1, 1, W)
        out_w = torch.cat((out_w1, out_w2_split), dim=1)
        out_w = self.fc1_w(out_w)
        out_w = self.relu(out_w)
        out_w = self.fc2_w(out_w)
        a_h = out_h.sigmoid()
        a_w = out_w.sigmoid()
        out_ca = x * a_w * a_h
        x_q = torch.split(out_ca, self.section, 1)
        x_q1 = self.conv1(x_q[0])
        x_q2 = self.conv3(self.avg_pool_3(x_q[1]) + x_q1)
        x_q3 = self.conv5(self.avg_pool_5(x_q[2]) + x_q2)
        x_q4 = self.conv7(self.avg_pool_7(x_q[3]) + x_q3)
        out = torch.cat((x_q1, x_q2, x_q3, x_q4), dim=1)
        return out


class SISM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SISM, self).__init__()
        self.DSConv1 = DSConv(in_planes//2, out_planes, kernel_size=3)
        self.DSConv2 = DSConv(in_planes//2, out_planes, kernel_size=3)
        self.deconv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.longdependency = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=out_planes, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.Conv2d(out_planes, out_planes, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=out_planes, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.deconv_layer(x)
        branch = self.DSConv1(x)
        weight = F.interpolate(self.sigmoid(self.longdependency(F.avg_pool2d(x, kernel_size=2, stride=2))),
                            size=(branch.shape[-2], branch.shape[-1]),
                            mode='nearest')
        out = self.DSConv2(branch*weight)
        out = out + x
        return out


class DCT2D(nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size

    def dct_2d(self, x):
        x = x.detach().cpu().numpy()
        x_dct = scipy.fft.dct(scipy.fft.dct(x, type=2, norm='ortho', axis=-1), type=2, norm='ortho', axis=-2)
        return torch.tensor(x_dct, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.block_size == 0 and W % self.block_size == 0
        x = x.view(B, C, H // self.block_size, self.block_size, W // self.block_size, self.block_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(-1, self.block_size, self.block_size)
        x = self.dct_2d(x)
        x = x.view(B, C, (H // self.block_size) * (W // self.block_size), self.block_size ** 2)
        x = x.permute(0, 1, 3, 2).reshape(B, C * (self.block_size ** 2), H // self.block_size,
                                          W // self.block_size)
        return F.interpolate(x, size=(H, W), mode='nearest')


class DCTFeatureExtractor(nn.Module):
    def __init__(self, freq_mean, freq_std):
        super().__init__()
        self.dct = DCT2D()

        self.register_buffer('freq_mean', freq_mean.view(1, -1, 1, 1))
        self.register_buffer('freq_std', freq_std.view(1, -1, 1, 1))

    def freq_decompose(self, freq):
        freq_y = freq[:, :64]
        freq_cb = freq[:, 64:128]
        freq_cr = freq[:, 128:192]
        high = torch.cat([
            freq_y[:, 32:],
            freq_cb[:, 32:],
            freq_cr[:, 32:]
        ], dim=1)
        low = torch.cat([
            freq_y[:, :32],
            freq_cb[:, :32],
            freq_cr[:, :32]
        ], dim=1)
        return high, low

    def forward(self, x):
        freq = self.dct(x)
        freq = (freq - self.freq_mean) / self.freq_std
        freq = freq / 7.0
        return self.freq_decompose(freq)


class SqueezeExcitation(nn.Module):
    """
    Modified from the official implementation of [torchvision.ops.SqueezeExcitation]
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): reduction ratio
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            reduction: int = 4,
            activation=nn.ReLU,
            scale_activation=nn.Sigmoid,
            pool='avgpool'
    ):
        super(SqueezeExcitation, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.transition = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        if out_channels // reduction == 0:
            reduction = 1

        if pool == 'avgpool':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'maxpool':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            print('Parameter pool is not avgpool or maxpool')
            return
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.transition(x)
        scale = self._scale(x)
        return scale * x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, bias=True,
                 se_ratio=2):
        super(DepthwiseSeparableConv, self).__init__()

        if dilation != 1:
            padding = dilation

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   dilation=dilation,
                                   groups=in_channels, bias=False, stride=stride)
        self.bn = nn.BatchNorm2d(in_channels)
        self.se = SqueezeExcitation(in_channels=in_channels, out_channels=in_channels, reduction=se_ratio)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = torch.relu(self.bn(out))
        out = self.se(out)
        out = self.pointwise(out)
        return out


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, bias=True):
        super(DSConv, self).__init__()

        if dilation != 1:
            padding = dilation

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=False, stride=stride)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = torch.relu(self.bn(out))
        out = self.pointwise(out)
        return out


class HBGM(nn.Module):

    def __init__(self, in_channels):
        super(HBGM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.reconv = nn.Conv2d(in_channels, 1, 3, 1, 1)

    def forward(self, x, pred_mask, freq):
        residual = x
        freq = F.interpolate(freq, size=x.size()[-2:], mode='bilinear', align_corners=True)
        pred_mask = F.interpolate(pred_mask, size=x.size()[-2:], mode='bilinear', align_corners=True)
        pred_mask = torch.sigmoid(pred_mask)
        background_att = 1 - pred_mask
        background_x = x * background_att
        bgx = self.conv1(background_x)
        pred_feature = x * pred_mask
        pf = self.conv2(pred_feature)
        feature = freq * x
        fusion_feature1 = bgx * x
        fusion_feature2 = pf * x
        fusion_feature = torch.cat([feature, fusion_feature1, fusion_feature2], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)
        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map
        out = fusion_feature + residual
        e = self.reconv(out)
        return out, e
