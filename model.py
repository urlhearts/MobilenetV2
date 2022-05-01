import torch.nn as nn
import torch


#  2017年 由 google 团队提出的
#  亮点 Depthwise Convolution (大大减少运算量和参数数量)
# 增加超参数α (控制卷积核个数)   和 β (控制图像大小)
# 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定，


# 定义超参数α  // 卷积核个数的倍率
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)

    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 定义BN结构
class ConvBNReLU(nn.Sequential):
    # 这里group 如果等于 in_channel 则是DW卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # “python中“//”是一个算术运算符,表示整数除法,它可以返回商的整数部分(向下取整
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(True)
        )


# 定义倒残差结构
class InvertedResNet(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResNet, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.short_cut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        # extend 可批量填入很多元素  跟append效果一样
        layers.extend([
            # 这里group 如果等于 in_channel 则是DW卷积
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
            # 不添加激活函数就等于 添加线性激活函数 Liner
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.short_cut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNet, self).__init__()
        block = InvertedResNet
        # 这里是把 卷积和的个数 设置为 round_nearest 的整数倍
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        Inverted_resdual_setting = [
            # t,c,n,s     t 是 扩展因子  使用1*1卷积扩充特征图个数   / c是输出的卷积核个数  / n是 bottleneck 重复的次数   / s是步距
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = []
        features.append(ConvBNReLU(3, input_channel, stride=2))
        for t, c, n, s in Inverted_resdual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            # 重复几次倒残差结构
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        # 初始化参数
        for m in self.modules():  # 会遍历我们的每一层
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 如果是一个全连接层 把我们权重调整成均值为0 方差为0.01的正态分布
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # 方差设置为1 均值设置为0
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)

        return x
