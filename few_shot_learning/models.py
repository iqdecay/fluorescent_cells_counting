import torch.nn as nn
import torch.nn.init as init
import torch


class FCRN(nn.Module):
    """Fully Convolutional Regression Network."""

    @staticmethod
    def _conv_downsampling(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        affine: bool = False,
    ):
        """Downsampling convulation layer."""
        layer = []
        layer.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=1, bias=False
            )
        )
        layer.append(nn.BatchNorm2d(out_channels, affine=affine))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    @staticmethod
    def _conv_upsampling(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        affine: bool = False,
    ):
        """Upsampling convulation layer."""
        layer = []
        layer.append(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=2, bias=False
            )
        )
        layer.append(nn.BatchNorm2d(out_channels, affine=affine))
        layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layer)

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        kernel_size: int = 3,
        add_sigmoid: bool = False,
        affine: bool = False,
    ):
        super(FCRN, self).__init__()
        self.add_sigmoid = add_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2, 2)

        # Encoder block
        self.conv_down_1 = self._conv_downsampling(
            in_channels, out_channels, kernel_size, affine=affine
        )
        self.conv_down_2 = self._conv_downsampling(
            out_channels, out_channels * 2, kernel_size, affine=affine
        )
        self.conv_down_3 = self._conv_downsampling(
            out_channels * 2, out_channels * 4, kernel_size, affine=affine
        )
        # Latent space
        self.conv_down_4 = self._conv_downsampling(
            out_channels * 4, out_channels * 16, kernel_size, affine=affine
        )
        # Decoder block
        self.conv_up_1 = self._conv_upsampling(
            out_channels * 16, out_channels * 4, 2, affine=affine
        )
        self.conv_up_2 = self._conv_upsampling(
            out_channels * 4, out_channels * 2, 2, affine=affine
        )
        self.conv_up_3 = self._conv_upsampling(
            out_channels * 2, out_channels, 2, affine=affine
        )
        self.conv_up_4 = nn.Conv2d(out_channels, in_channels, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights of each layer."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """Peform forward propagation."""
        x = self.maxpool(self.conv_down_1(x))
        x = self.maxpool(self.conv_down_2(x))
        x = self.maxpool(self.conv_down_3(x))

        x = self.conv_down_4(x)
        feature_distill = x

        x = self.conv_up_1(x)
        x = self.conv_up_2(x)
        x = self.conv_up_3(x)
        x = self.conv_up_4(x)

        if self.add_sigmoid:
            out = nn.Sigmoid(x)
        else:
            out = x

        return out, feature_distill


class UNet(nn.Module):
    """U-Net."""

    @staticmethod
    def _conv_double(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        affine: bool = False,
    ):
        """Double convolutional layer."""
        layer = []

        layer.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        )
        layer.append(nn.BatchNorm2d(out_channels, affine=affine))
        layer.append(nn.ReLU(inplace=True))

        layer.append(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        )
        layer.append(nn.BatchNorm2d(out_channels, affine=affine))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    def __init__(
        self,
        n_class: int = 1,
        in_channels: int = 1,
        out_channels: int = 32,
        kernel_size: int = 3,
        add_sigmoid: bool = False,
        affine: bool = False,
    ):
        super(UNet, self).__init__()
        self.add_sigmoid = add_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2, 2)

        # Encoder block
        self.conv_down_1 = self._conv_double(
            in_channels, out_channels, kernel_size, affine=affine
        )
        self.conv_down_2 = self._conv_double(
            out_channels, out_channels * 2, kernel_size, affine=affine
        )
        self.conv_down_3 = self._conv_double(
            out_channels * 2, out_channels * 4, kernel_size, affine=affine
        )
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        # Decoder block
        self.conv_up_1 = self._conv_double(
            out_channels * 6, out_channels * 2, kernel_size, affine=affine
        )
        self.conv_up_2 = self._conv_double(
            out_channels * 3, out_channels, kernel_size, affine=affine
        )
        self.conv_up_3 = nn.Conv2d(out_channels, n_class, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights of each layer."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """Peform forward propagation."""
        conv_down_1 = self.conv_down_1(x)
        x = self.maxpool(conv_down_1)

        conv_down_2 = self.conv_down_2(x)
        x = self.maxpool(conv_down_2)

        x = self.conv_down_3(x)
        feature_distill = x

        x = self.upsample(x)
        x = torch.cat([x, conv_down_2], dim=1)
        x = self.conv_up_1(x)

        x = self.upsample(x)
        x = torch.cat([x, conv_down_1], dim=1)
        x = self.conv_up_2(x)
        x = self.conv_up_3(x)

        if self.add_sigmoid:
            out = self.sigmoid(x)
        else:
            out = x

        return out, feature_distill
