import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from timm.models.layers import  DropPath
from utils import GRN


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, k_size=3, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=k_size//2, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(ConvBlock):
    expansion = 1  # 扩展系数

    def __init__(self, in_channels, out_channels, stride=1, k_size=3, dilation=1):
        super(ResidualBlock, self).__init__(in_channels, out_channels, stride, k_size, dilation)
        p = k_size // 2 * dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=p, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.the_bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=k_size, stride=1, padding=p, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return nn.ReLU()(x)

class RecurrentBlock(nn.Module):
    def __init__(self, out_ch, k_size, t=2, groups=1):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, stride=1, padding=k_size//2, bias=False, groups=groups),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(ConvBlock):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_channels, out_channels, stride, k_size, dilation):
        super(RRCNNBlock, self).__init__(in_channels, out_channels, stride, k_size, dilation)
        assert dilation == 1
        t = 2
        self.RCNN = nn.Sequential(
            RecurrentBlock(out_channels, k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            RecurrentBlock(out_channels, k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return nn.ReLU()(out)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def expend_as(tensor, rep):
    return tensor.repeat(1, rep, 1, 1)



class Channelblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Channelblock, self).__init__()
        self.conv1 = nn.Sequential(
            RecurrentBlock(in_channels, k_size=3, t=2, groups=in_channels),  # Depthwise Convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Pointwise Convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            RecurrentBlock(in_channels, k_size=5, t=2, groups=in_channels),  # Depthwise Convolution
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)

        combined = torch.cat([conv1, conv2], dim=1)
        pooled = self.global_avg_pool(combined)
        pooled = torch.flatten(pooled, 1)
        sigm = self.fc(pooled)

        a = sigm.view(-1, sigm.size(1), 1, 1)
        a1 = 1 - sigm
        a1 = a1.view(-1, a1.size(1), 1, 1)

        y = conv1 * a
        y1 = conv2 * a1

        combined = torch.cat([y, y1], dim=1)
        out = self.conv3(combined)

        return out




class Spatialblock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Spatialblock, self).__init__()
        self.conv1 = nn.Sequential(
            RecurrentBlock(in_channels, k_size=3, t=2, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            RecurrentBlock(out_channels, k_size=5, t=2, groups=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=size, padding=(size // 2)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x, channel_data):
        conv1 = self.conv1(x)
        spatil_data = self.conv2(conv1)

        data3 = torch.add(channel_data, spatil_data)
        data3 = torch.relu(data3)
        data3 = nn.Conv2d(data3.size(1), 1, kernel_size=1, padding=0).cuda()(data3)
        data3 = torch.sigmoid(data3)

        a = expend_as(data3, channel_data.size(1))
        y = a * channel_data

        a1 = 1 - data3
        a1 = expend_as(a1, spatil_data.size(1))
        y1 = a1 * spatil_data

        combined = torch.cat([y, y1], dim=1)
        out = self.final_conv(combined)

        return out



class DWAM(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        super(DWAM,self).__init__()
        self.channel_block = Channelblock(in_channels, out_channels)
        self.spatial_block = Spatialblock(out_channels, out_channels, size)

    def forward(self, x):
        channel_data = self.channel_block(x)
        dwam_data = self.spatial_block(x, channel_data)
        return dwam_data
class RecurrentConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwam=DWAM(in_channels=32,out_channels=32)
        self.dwconv = RecurrentBlock(dim, k_size=7, groups=dim)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = RecurrentBlock(dim, k_size=3)
        # self.pwconv1 = nn.Conv2d(dim,4* dim, kernel_size=3, padding=1)# nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(10 * dim)
        self.pwconv2 = RecurrentBlock(dim, k_size=3)
        # self.pwconv2 = nn.Conv2d(4*dim, dim, kernel_size=3, padding=1)#nn.Linear(4 * dim, dim)
        #self.cacp = CACPBlock(ch_in=32, ch_out=32)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.cacp = CACPBlock(ch_in=32, ch_out=32)
        #self.eff = EFF(in_dim=dim, out_dim=dim, is_bottom=True)
        #self.hham = HAAM(in_channels=32, out_channels=32)
    def forward(self, x):

        input = x
        x=self.dwam(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        #x = self.cacp(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        #print(x.shape)
        #x=self.cacp(x)
        #x=self.eff(input,self.drop_path(x))
        return x


class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class CACPBlock(nn.Module):

    def __init__(self, ch_in, ch_out, channelAttention_reduce=4):
        super().__init__()

        self.C = ch_in
        self.O = ch_out


        self.ca = ChannelAttention(input_channels=ch_in, internal_neurons=ch_in // channelAttention_reduce)
        # other initialization code here
        self.dconv5_5 = nn.Conv2d(ch_in, ch_in, kernel_size=5, padding=2, groups=ch_in)
        self.dconv1_7 = nn.Conv2d(ch_in, ch_in, kernel_size=(7, 1), padding=(3, 0), groups=ch_in)
        self.dconv7_1 = nn.Conv2d(ch_in, ch_in, kernel_size=(7, 1), padding=(3, 0), groups=ch_in)
        self.dconv1_11 = nn.Conv2d(ch_in, ch_in, kernel_size=(1, 11), padding=(0, 5), groups=ch_in)
        self.dconv11_1 = nn.Conv2d(ch_in, ch_in, kernel_size=(11, 1), padding=(5, 0), groups=ch_in)
        self.dconv1_21 = nn.Conv2d(ch_in, ch_in, kernel_size=(1, 21), padding=(0, 10), groups=ch_in)
        self.dconv21_1 = nn.Conv2d(ch_in, ch_in, kernel_size=(21, 1), padding=(10, 0), groups=ch_in)
        self.conv = nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron

        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out
class FFNBlock2(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels,hidden_features,kernel_size=(1,1),padding=0)
        self.conv2 = nn.Conv2d(hidden_channels,out_features,kernel_size=(1,1),padding=0)
        self.dconv = nn.Conv2d(hidden_features,hidden_features,kernel_size=(3,3),padding=(1,1),groups=hidden_features)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dconv(x)
        x = self.act(x)
        x = self.conv2(x)
        return x
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.,ffn_expand=4, channelAttention_reduce=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.bn = nn.BatchNorm2d(num_features=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_block = FFNBlock2(dim,dim*ffn_expand)
        self.repmlp_block = CACPBlock(ch_in=dim, ch_out=dim, channelAttention_reduce=channelAttention_reduce)
        #self.eff = EFF(in_dim=dim, out_dim=dim, is_bottom=True)
    def forward(self, x):
        input = x.clone()

        x = self.bn(x)
        x = self.repmlp_block(x)
        #x = self.eff(self.drop_path(x), input)
        x = input + self.drop_path(x)
        x2 = self.bn(x)
        x2 = self.ffn_block(x2)
        #x = self.eff(self.drop_path(x2), x)
        x = x + self.drop_path(x2)

        return x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class Efficient_Attention_Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=32):
        super(Efficient_Attention_Gate, self).__init__()
        self.num_groups = num_groups
        self.grouped_conv_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )

        self.grouped_conv_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        psi = self.psi(self.relu(x1 + g1))
        out = x * psi
        out += x

        return out
class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)

        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return v
class EFF(nn.Module):
    def __init__(self, in_dim, out_dim, is_bottom=False):
        super().__init__()
        self.is_bottom = is_bottom
        if not is_bottom:
            self.EAG = Efficient_Attention_Gate(in_dim, in_dim, out_dim)
        else:
            self.EAG = nn.Identity()
        self.ECA = EfficientChannelAttention(in_dim*2)
        self.SA = SpatialAttention()

    def forward(self, x, skip):
        if not self.is_bottom:
            EAG_skip = self.EAG(x, skip)
            x = torch.cat((EAG_skip, x), dim=1)
        else:
            x = self.EAG(x)
        x = self.ECA(x) * x
        x = self.SA(x) * x
        return x

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x







if __name__ == '__main__':
    pp=RecurrentConvNeXtBlock()
    print(pp)


