"""Unified multi-task model."""

import os

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder


class _CheckpointClassifier(nn.Module):
    """Classifier architecture compatible with classifier.pth."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3):
        super(_CheckpointClassifier, self).__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Identity(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class _CheckpointLocalizer(nn.Module):
    """Localization architecture compatible with localizer.pth."""

    def __init__(self, in_channels: int = 3):
        super(_CheckpointLocalizer, self).__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.localization_head = nn.Sequential(
            nn.Identity(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.localization_head(x)


class _DecodeBlock(nn.Module):
    """Decoder block whose parameter names match the saved U-Net checkpoint."""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int):
        super(_DecodeBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class _CheckpointUNet(nn.Module):
    """Segmentation architecture compatible with unet.pth."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super(_CheckpointUNet, self).__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.decode4 = _DecodeBlock(512, 512, 512)
        self.decode3 = _DecodeBlock(512, 256, 256)
        self.decode2 = _DecodeBlock(256, 128, 128)
        self.decode1 = _DecodeBlock(128, 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, skips = self.encoder(x, return_features=True)
        x = skips["s5"]
        x = self.decode4(x, skips["s4"])
        x = self.decode3(x, skips["s3"])
        x = self.decode2(x, skips["s2"])
        x = self.decode1(x, skips["s1"])
        return self.final_conv(x)


class MultiTaskPerceptionModel(nn.Module):
    """Unified wrapper over the three trained task-specific checkpoints."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
        num_classes: int = None,
    ):
        """
        Initialize the multi-task model using the three trained checkpoints.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super(MultiTaskPerceptionModel, self).__init__()

        if num_classes is not None:
            num_breeds = num_classes

        os.makedirs("checkpoints", exist_ok=True)
        classifier_path = os.path.join("checkpoints", os.path.basename(classifier_path))
        localizer_path = os.path.join("checkpoints", os.path.basename(localizer_path))
        unet_path = os.path.join("checkpoints", os.path.basename(unet_path))

        import gdown
        if not os.path.exists(classifier_path):
            gdown.download(
                id="1JgctJgD9EP8PgL8--0kGrKhfRCWdhRlA",
                output=classifier_path,
                quiet=False,
            )
        if not os.path.exists(localizer_path):
            gdown.download(
                id="1M7Lp9zrOneDXCcxlB3-8JNJwJl7zKhCM",
                output=localizer_path,
                quiet=False,
            )
        if not os.path.exists(unet_path):
            gdown.download(
                id="13pAD3ziJMXZAzwWxFSPjQlA6VvvE2-6D",
                output=unet_path,
                quiet=False,
            )

        self.classifier = _CheckpointClassifier(
            num_classes=num_breeds,
            in_channels=in_channels,
        )
        self.localizer = _CheckpointLocalizer(in_channels=in_channels)
        self.segmenter = _CheckpointUNet(
            num_classes=seg_classes,
            in_channels=in_channels,
        )

        self._load_model_weights(self.classifier, classifier_path)
        self._load_model_weights(self.localizer, localizer_path)
        self._load_model_weights(self.segmenter, unet_path)

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            elif "weights" in checkpoint:
                checkpoint = checkpoint["weights"]

        cleaned_state_dict = {}
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned_state_dict[key] = value

        return cleaned_state_dict

    def _load_model_weights(self, model: nn.Module, checkpoint_path: str):
        state_dict = self._load_checkpoint(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        classification = self.classifier(x)
        localization = self.localizer(x)
        segmentation = self.segmenter(x)

        height = x.shape[-2]
        width = x.shape[-1]
        localization = localization.clone()
        localization[:, 0] = localization[:, 0] * width
        localization[:, 1] = localization[:, 1] * height
        localization[:, 2] = localization[:, 2] * width
        localization[:, 3] = localization[:, 3] * height

        return {
            "classification": classification,
            "localization": localization,
            "segmentation": segmentation,
        }


MultiTaskVGG = MultiTaskPerceptionModel
