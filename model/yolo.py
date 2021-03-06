import numpy as np
import torch
import  torch.nn as nn

class ConvBlock(nn.Module):
    """ 卷积块 """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.features(x)

class ResidualUnit(nn.Module):
    """ 残差单元 """
    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, in_channels//2, 1, padding=0),
            ConvBlock(in_channels//2, in_channels, 3),
        )

    def forward(self, x):
        y = self.features(x)
        return x+y

class ResidualBlock(nn.Module):
    """ 残差块 """
    def __init__(self, in_channels: int, n_residuals=1):
        """
        Parameters
        ----------
        in_channels: int
            输入通道数

        n_residuals: int
            残差单元的个数
        """
        super().__init__()
        self.conv = ConvBlock(in_channels, in_channels*2, 3, stride=2)
        self.residual_units = nn.Sequential(*[
            ResidualUnit(2*in_channels) for _ in range(n_residuals)
        ])

    def forward(self, x):
        return self.residual_units(self.conv(x))

class Darknet(nn.Module):
    """ 主干网络 """

    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(3, 32, 3)
        self.residuals = nn.ModuleList([
            ResidualBlock(32, 1),
            ResidualBlock(64, 2),
            ResidualBlock(128, 8),
            ResidualBlock(256, 8),
            ResidualBlock(512, 4),
        ])

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            输入图像

        Returns
        -------
        x1: Tensor of shape `(N, 1024, H/32, W/32)`
        x2: Tensor of shape `(N, 512, H/16, W/16)`
        x3: Tensor of shape `(N, 256, H/8, W/8)`
        """
        x3 = self.conv(x)
        for layer in self.residuals[:-2]:
            x3 = layer(x3)

        x2 = self.residuals[-2](x3)
        x1 = self.residuals[-1](x2)
        return x1, x2, x3

class YoloBlock(nn.Module):
    """ Yolo 块 """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.features = nn.Sequential(*[
            ConvBlock(in_channels, out_channels, 1, padding=0),
            ConvBlock(out_channels, out_channels*2, 3, padding=1),
            ConvBlock(out_channels*2, out_channels, 1, padding=0),
            ConvBlock(out_channels, out_channels*2, 3, padding=1),
            ConvBlock(out_channels*2, out_channels, 1, padding=0),
        ])

    def forward(self, x):
        return self.features(x)

class Yolo(nn.Module):
    """ Yolo 神经网络 """

    def __init__(self, n_classes: int, image_size=416, anchors: list = None, nms_thresh=0.45):
        """
        Parameters
        ----------
        n_classes: int
            类别数

        image_size: int
            图片尺寸，必须是 32 的倍数

        anchors: list
            输入图像大小为 416 时对应的先验框

        nms_thresh: float
            非极大值抑制的交并比阈值，值越大保留的预测框越多
        """
        super().__init__()
        if image_size <= 0 or image_size % 32 != 0:
            raise ValueError("image_size 必须是 32 的倍数")

        # 先验框
        anchors = anchors or [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]]
        ]
        anchors = np.array(anchors, dtype=np.float32)
        anchors = anchors*image_size/416
        self.anchors = anchors.tolist()

        self.n_classes = n_classes
        self.image_size = image_size

        self.darknet = Darknet()
        self.yolo1 = YoloBlock(1024, 512)
        self.yolo2 = YoloBlock(768, 256)
        self.yolo3 = YoloBlock(384, 128)
        # YoloBlock 后面的卷积部分
        out_channels = (n_classes+5)*3
        self.conv1 = nn.Sequential(*[
            ConvBlock(512, 1024, 3),
            nn.Conv2d(1024, out_channels, 1)
        ])
        self.conv2 = nn.Sequential(*[
            ConvBlock(256, 512, 3),
            nn.Conv2d(512, out_channels, 1)
        ])
        self.conv3 = nn.Sequential(*[
            ConvBlock(128, 256, 3),
            nn.Conv2d(256, out_channels, 1)
        ])
        # 上采样
        self.upsample1 = nn.Sequential(*[
            nn.Conv2d(512, 256, 1),
            nn.Upsample(scale_factor=2)
        ])
        self.upsample2 = nn.Sequential(*[
            nn.Conv2d(256, 128, 1),
            nn.Upsample(scale_factor=2)
        ])

        # 探测器，用于处理输出的特征图，后面会讲到
        #self.detector = Detector(
        #   self.anchors, image_size, n_classes, conf_thresh=0.1, nms_thresh=nms_thresh)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            输入图像

        Returns
        -------
        y1: Tensor of shape `(N, 255, H/32, W/32)`
            最小的特征图

        y2: Tensor of shape `(N, 255, H/16, W/16)`
            中等特征图

        y3: Tensor of shape `(N, 255, H/8, W/8)`
            最大的特征图
        """
        x1, x2, x3 = self.darknet(x)
        x1 = self.yolo1(x1)
        y1 = self.conv1(x1)

        x2 = self.yolo2(torch.cat([self.upsample1(x1), x2], 1))
        y2 = self.conv2(x2)

        x3 = self.yolo3(torch.cat([self.upsample2(x2), x3], 1))
        y3 = self.conv3(x3)

        return y1, y2, y3




