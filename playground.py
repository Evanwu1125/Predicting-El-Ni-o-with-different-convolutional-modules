import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import torch.nn.functional as F

'''1.第一个模块：朴素卷积'''
## 纯卷积神经网络
class ConvNetwork(nn.Module):
    def __init__(self, M_Num, N_Num):
        self.M = M_Num
        self.N = N_Num
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, M_Num, kernel_size=(4, 8), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(stride=(2, 2), kernel_size=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), stride=(1, 1), padding="same"),
            nn.Tanh(), )
        self.dense = nn.Sequential(
            nn.Linear(6 * 18 * M_Num, N_Num),
            nn.Linear(N_Num, 23))

    def forward(self, InData):
        x = self.conv(InData)
        x = x.reshape(-1, 6 * 18 * self.M)
        x = self.dense(x)
        return x

'''2.第二个模块：加了残差块'''
## 残差网络
class ResNet(nn.Module):
    def __init__(self, M_Num, N_Num):
        self.M = M_Num
        self.N = N_Num
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, M_Num, kernel_size=(4, 8), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(stride=(2, 2), kernel_size=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), stride=(1, 1), padding="same"),
            nn.Tanh(), )
        #这里是我定义的残差结构
        self.resconnenction = nn.Sequential(
            nn.Conv2d(6, M_Num, kernel_size = (3, 3), padding = 'same'),
            nn.ReLU(),
            nn.Conv2d(M_Num, M_Num, kernel_size = (3,3), padding = 'same'),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(6 * 18 * M_Num, N_Num),
            nn.Linear(N_Num, 23))

    def forward(self, InData):
        x = self.conv(InData)
        #残差连接
        res_X = self.conv(InData)
        x += res_X
        x = x.reshape(-1, 6 * 18 * self.M)
        x = self.dense(x)
        return x

'''3.第三个模块：SEnet'''
## SEnet
#先定义SELayer的模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SENet(nn.Module):
    def __init__(self, M_Num, N_Num):
        self.M = M_Num
        self.N = N_Num
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, M_Num, kernel_size=(4, 8), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(stride=(2, 2), kernel_size=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), stride=(1, 1), padding="same"),
            nn.Tanh(), )
        self.se = SELayer(M_Num)
        self.dense = nn.Sequential(
            nn.Linear(6 * 18 * M_Num, N_Num),
            nn.Linear(N_Num, 23))

    def forward(self, InData):
        x = self.conv(InData)
        new_x = x
        new_x = self.se(new_x)
        x += new_x
        x = x.reshape(-1, 6 * 18 * self.M)
        x = self.dense(x)
        return x

'''4.第四个模块：CBAM模块'''
## CBAM模块

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out

class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp

class CBAMNet(nn.Module):
    def __init__(self, M_Num, N_Num):
        self.M = M_Num
        self.N = N_Num
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, M_Num, kernel_size=(4, 8), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(stride=(2, 2), kernel_size=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), stride=(1, 1), padding="same"),
            nn.Tanh(), )
        self.cbam = CBAM(n_channels_in=M_Num, reduction_ratio=2, kernel_size= 3 )
        self.dense = nn.Sequential(
            nn.Linear(6 * 18 * M_Num, N_Num),
            nn.Linear(N_Num, 23))

    def forward(self, InData):
        x = self.conv(InData)
        new_x = x
        new_x = self.cbam(new_x)
        x += new_x
        x = x.reshape(-1, 6 * 18 * self.M)
        x = self.dense(x)
        return x


'''5.第五个模块：BAM模块'''
## BAM网络
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)

def conv7x7(in_channels, out_channels, stride=1, padding=3, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=padding, dilation=dilation, bias=False)

class BAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio, dilation):
        super(BAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(in_features=in_channel, out_features=self.hid_channel)
        self.bn1_1d = nn.BatchNorm1d(self.hid_channel)
        self.fc2 = nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        self.bn2_1d = nn.BatchNorm1d(in_channel)

        self.conv1 = conv1x1(in_channel, self.hid_channel)
        self.bn1_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv2 = conv3x3(self.hid_channel, self.hid_channel, stride=1, padding=self.dilation, dilation=self.dilation)
        self.bn2_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv3 = conv3x3(self.hid_channel, self.hid_channel, stride=1, padding=self.dilation, dilation=self.dilation)
        self.bn3_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv4 = conv1x1(self.hid_channel, 1)
        self.bn4_2d = nn.BatchNorm2d(1)

    def forward(self, x):
        # Channel attention
        Mc = self.globalAvgPool(x)
        Mc = Mc.view(Mc.size(0), -1)

        Mc = self.fc1(Mc)
        Mc = self.bn1_1d(Mc)
        Mc = self.relu(Mc)

        Mc = self.fc2(Mc)
        Mc = self.bn2_1d(Mc)
        Mc = self.relu(Mc)

        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)

        # Spatial attention
        Ms = self.conv1(x)
        Ms = self.bn1_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv2(Ms)
        Ms = self.bn2_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv3(Ms)
        Ms = self.bn3_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv4(Ms)
        Ms = self.bn4_2d(Ms)
        Ms = self.relu(Ms)

        Ms = Ms.view(x.size(0), 1, x.size(2), x.size(3))
        Mf = 1 + self.sigmoid(Mc * Ms)
        return x * Mf

class BAMNet(nn.Module):
    def __init__(self, M_Num, N_Num):
        self.M = M_Num
        self.N = N_Num
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, M_Num, kernel_size=(4, 8), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), padding="same"),
            nn.Tanh(),
            nn.MaxPool2d(stride=(2, 2), kernel_size=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), stride=(1, 1), padding="same"),
            nn.Tanh(), )
        # self.cbam = CBAM(n_channels_in=M_Num, reduction_ratio=2, kernel_size= 3 )
        self.bam = BAM(in_channel= M_Num, reduction_ratio=2, dilation= 1)
        self.dense = nn.Sequential(
            nn.Linear(6 * 18 * M_Num, N_Num),
            nn.Linear(N_Num, 23))

    def forward(self, InData):
        x = self.conv(InData)
        new_x = x
        new_x = self.bam(new_x)
        x += new_x
        x = x.reshape(-1, 6 * 18 * self.M)
        x = self.dense(x)
        return x

#这里举一个小小的例子方便大家理解网络运行
if __name__ == '__main__':
    #这里我们生成一个通道数为6， 长和宽为24*72的数据
    #通道数为6的原因：因为我们在进行训练的时候是把3个SST和3个HT数据拼接到了一起
    #长和宽为24*72的原因：因为经度和纬度上的数据范围分别是24和72
    #所以简单来说，我们的数据是由6组大数据构成，每组大数据又是由24条纬度和72条经度上的数据构成的，所以是6 * 24 * 72
    dummy_input = torch.rand(6, 24, 72)
    #这里M_num和N_num是卷积过程和全连接过程中通道的转化数，可以随便设，之所以设置成30是和源码保持一致
    M_num = 30;N_num = 30
    '''选择不同的网络'''
    #net = SEnet(30,30) #SEnet
    #net = ConvNetwork(30, 30) #纯卷积
    #net = CBAMNet(30, 30) #CBAM网络
    #net = ResNet(30, 30) #残差
    net = BAMNet(30, 30)
    #这里我们把批量设置成2，因为BacthNormalization只适用于批量大于一的数据
    input_data = torch.rand(2, 6, 24, 72)#生成对应输入形状的随机数据
    print(f"input_data.shape:{input_data.shape}")
    output_data = net(input_data)
    print(f"output_data.shape:{output_data.shape}")