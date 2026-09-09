"""
Fast-SCNN: Fast Segmentation Convolutional Neural Network
Tailored for real-time off-road terrain traversability segmentation on edge devices (Jetson Orin).

Reference:
Poudel et al., "Fast-SCNN: Fast Segmentation Convolutional Neural Network", BMVC 2019.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Standard Convolution + BatchNorm + ReLU module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        relu: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (Depthwise 3x3 + Pointwise 1x1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        relu6: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn_dw(self.depthwise(x)))
        x = self.act(self.bn_pw(self.pointwise(x)))
        return x


class InvertedResidual(nn.Module):
    """Inverted Residual Bottleneck Block (MobileNetV2 style)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(
                ConvBNReLU(
                    in_channels, hidden_dim, kernel_size=1, stride=1, padding=0
                )
            )

        # Depthwise conv
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # Linear pointwise projection (no activation)
                nn.Conv2d(
                    hidden_dim, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class LearningToDownsample(nn.Module):
    """
    Learning to Downsample (LDS) block:
    Fast, efficient 8x spatial downsampling preserving low-level spatial geometry.
    Input: (B, 3, H, W) -> Output: (B, 64, H/8, W/8)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = ConvBNReLU(
            in_channels, 32, kernel_size=3, stride=2, padding=1
        )
        self.dsconv1 = DepthwiseSeparableConv(32, 48, stride=2)
        self.dsconv2 = DepthwiseSeparableConv(48, 64, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM):
    Gathers global contextual information at multiple spatial grid scales.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bin_sizes: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    ConvBNReLU(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0
                    ),
                )
                for bin_size in bin_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        pyramid_features = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(
                pooled, size=(h, w), mode="bilinear", align_corners=False
            )
            pyramid_features.append(upsampled)
        return torch.cat(pyramid_features, dim=1)


class GlobalFeatureExtractor(nn.Module):
    """
    Global Feature Extractor (GFE):
    Captures deep semantic context using inverted bottleneck residuals and PPM.
    Input: (B, 64, H/8, W/8) -> Output: (B, 256, H/32, W/32)
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        # Bottleneck stage 1: downsample by 2x -> H/16, W/16
        self.block1 = nn.Sequential(
            InvertedResidual(in_channels, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
        )
        # Bottleneck stage 2: downsample by 2x -> H/32, W/32
        self.block2 = nn.Sequential(
            InvertedResidual(64, 96, stride=2, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
        )
        # Bottleneck stage 3: stride 1 -> H/32, W/32
        self.block3 = nn.Sequential(
            InvertedResidual(96, 128, stride=1, expand_ratio=6),
            InvertedResidual(128, 128, stride=1, expand_ratio=6),
            InvertedResidual(128, 128, stride=1, expand_ratio=6),
        )
        # PPM: 128 + 4 * 32 = 256 channels
        self.ppm = PyramidPoolingModule(128, 32, bin_sizes=(1, 2, 4, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module (FFM):
    Fuses shallow high-resolution spatial features from LDS (H/8)
    with deep contextual semantics from GFE (H/32).
    Output: (B, 128, H/8, W/8)
    """

    def __init__(
        self,
        high_res_channels: int = 64,
        low_res_channels: int = 256,
        out_channels: int = 128,
    ):
        super().__init__()
        # Deep path 1x1 conv
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_res_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Shallow path 1x1 conv
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_res_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, high_res: torch.Tensor, low_res: torch.Tensor
    ) -> torch.Tensor:
        h, w = high_res.shape[2], high_res.shape[3]
        low_res_up = F.interpolate(
            self.conv_low(low_res), size=(h, w), mode="bilinear", align_corners=False
        )
        high_res_proj = self.conv_high(high_res)
        return self.relu(high_res_proj + low_res_up)


class FastSCNN(nn.Module):
    """
    Fast-SCNN Model for real-time semantic segmentation.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        num_classes: Number of semantic classes (default: 4 for off-road traversability:
                     0: Background/Sky, 1: Traversable, 2: Positive Obstacle, 3: Negative Obstacle).
        dropout_rate: Classifier dropout rate (default: 0.1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 1. Learning to Downsample (LDS)
        self.lds = LearningToDownsample(in_channels=in_channels)

        # 2. Global Feature Extractor (GFE)
        self.gfe = GlobalFeatureExtractor(in_channels=64)

        # 3. Feature Fusion Module (FFM)
        self.ffm = FeatureFusionModule(
            high_res_channels=64, low_res_channels=256, out_channels=128
        )

        # 4. Classifier Head
        self.classifier = nn.Sequential(
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 128, stride=1),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: (B, C, H, W)
        Output: (B, num_classes, H, W)
        """
        input_size = (x.shape[2], x.shape[3])

        # Step 1: Extract shallow high-resolution features (H/8, W/8)
        high_res = self.lds(x)

        # Step 2: Extract deep global semantic features (H/32, W/32)
        low_res = self.gfe(high_res)

        # Step 3: Fuse features (H/8, W/8)
        fused = self.ffm(high_res, low_res)

        # Step 4: Classification head and 8x upsampling to input resolution
        logits = self.classifier(fused)
        output = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )
        return output


if __name__ == "__main__":
    model = FastSCNN(in_channels=3, num_classes=4)
    x = torch.randn(2, 3, 512, 1024)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
