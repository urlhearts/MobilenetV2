from typing import Optional, List, Callable

import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import functional as F
from functools import partial
import numpy
import torchvision.models.mobilenetv3


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


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 # 卷积后
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 # 卷积函数
                 activation_layers: Optional[Callable[..., nn.Module]] = None
                 ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layers is None:
            activation_layers = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes,
                      out_channels=out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False
                      ),
            norm_layer(out_planes),
            activation_layers(inplace=True)
        )


# SE 注意力机制
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        # 这里我错误的使用了max pool 导致在训练的时候 一轮迁移学习效果0.3
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        # 这里的scale 是全连接层的数据 我们这里需要把他与输入进来的x 卷积核的 参数进行相乘
        return scale * x


# 每一层的参数
class InvertedResidualConfig(nn.Module):
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,  # 第一层卷积核使用的个数
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float,  # 调节我们每一层 卷积核的倍率因子
                 ):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se  # 是否使用se模块
        self.use_hs = activation == 'HS'  # 是否使用
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


# ---------------------------------------------------------------- 劳动节开始工作~
# ---------------------------------------------------------------- 劳动节开始工作~
# ---------------------------------------------------------------- 劳动节开始工作~

# 定义BN结构
# class ConvBNReLU(nn.Sequential):
#     # 这里group 如果等于 in_channel 则是DW卷积
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
#         # “python中“//”是一个算术运算符,表示整数除法,它可以返回商的整数部分(向下取整
#         padding = (kernel_size - 1) // 2
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU6(True)
#         )


# 定义倒残差结构
class InvertedResNet(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]
                 ):
        super(InvertedResNet, self).__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError('illegal stride value')
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []

        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layers=activation_layer))
        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.expanded_c,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,
                                       norm_layer=norm_layer,
                                       activation_layers=activation_layer
                                       ))
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layers=nn.Identity
                                       ))
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 # 传入我们参数的一系列列表
                 inverted_residual_setting: List[InvertedResidualConfig],
                 # 最后一层
                 last_channel: int,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(MobileNetV3, self).__init__()
        if not inverted_residual_setting:
            raise ValueError('the inverted_residual_setting is None')
        elif not (isinstance(inverted_residual_setting, List) and all(
                [isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be a List')

        if block is None:
            block = InvertedResNet

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []
        # 这里是把 卷积和的个数 设置为 round_nearest 的整数倍

        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layers=nn.Hardswish
                                       ))
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layers=nn.Hardswish,
                                       ))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_c, last_channel),
            nn.Hardswish(True),
            nn.Dropout(p=0.2, inplace=True),
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
        x = self.classifier(x)
        return x


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, 'RE', 1),
        bneck_conf(16, 3, 64, 24, False, 'RE', 2),
        bneck_conf(24, 3, 72, 24, False, 'RE', 1),
        bneck_conf(24, 5, 72, 40, True, 'RE', 2),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bneck_conf(40, 3, 240, 80, False, 'HS', 2),
        bneck_conf(80, 3, 200, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bneck_conf(80, 3, 480, 112, True, 'HS', 1),
        bneck_conf(112, 3, 672, 112, True, 'HS', 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, 'HS', 2),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes
                       )
