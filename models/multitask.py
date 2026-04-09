"""Unified multi-task model."""

import os

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import LocalizationModel
from .segmentation import UNetVGG11
from .vgg11 import VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Unified wrapper over the three trained task-specific models."""

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

        self.classifier = VGG11Classifier(num_classes=num_breeds)
        self.localizer = LocalizationModel(
            VGG11Encoder(in_channels=in_channels),
            freeze_early=False,
        )
        self.segmenter = UNetVGG11(
            VGG11Encoder(in_channels=in_channels),
            num_classes=seg_classes,
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
        model.load_state_dict(state_dict, strict=False)

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
