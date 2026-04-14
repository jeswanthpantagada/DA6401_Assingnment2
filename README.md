**wandb report link** :  https://wandb.ai/ee23b052-iitm-india/DA_DL_Ass2/reportlist



# DA6401 Assignment 2 — Visual Perception Pipeline

A complete multi-task deep learning pipeline built in PyTorch for the Oxford-IIIT Pet Dataset,
covering classification, object localization, and semantic segmentation under a unified architecture.

---

## Project Structure
da6401_assignment_2/
├── models/
│   ├── __init__.py
│   ├── layers.py               # Custom Dropout (inverted scaling, no nn.Dropout)
│   ├── vgg11.py                # VGG11 encoder with skip-connection support
│   ├── classification.py       # VGG11 + FC head (37-class breed classifier)
│   ├── localization.py         # VGG11 encoder + regression head (bbox)
│   ├── segmentation.py         # UNetVGG11 + DiceLoss + SegmentationLoss
│   └── multitask.py            # Unified MultiTaskPerceptionModel
├── losses/
│   ├── __init__.py
│   └── iou_loss.py             # Custom IoU Loss (no external libraries)
├── notebooks/
│   └── da6401_wandb_report.ipynb   # W&B report notebook (Kaggle-ready)
├── pets_dataset.py             # Oxford-IIIT PetDataset with albumentations
└── README.md
---

## Tasks Implemented

### Task 1 — VGG11 Classification (`classification.py`)
- VGG11 built from scratch using `torch.nn` primitives only
- `BatchNorm2d` after every `Conv2d`, before `ReLU` — reduces internal covariate shift and enables higher stable learning rates
- `CustomDropout` (p=0.5) in FC layers only — spatial weight sharing makes dropout ineffective in conv layers; FC dropout prevents co-adaptation and creates an implicit ensemble
- Output: 37-class breed logits

### Task 2 — Object Localization (`localization.py`)
- VGG11 convolutional backbone reused as encoder
- **Partial fine-tuning** strategy: blocks 1–3 frozen (generic edges/textures), blocks 4–5 unfrozen (task-specific spatial features)
- Regression head outputs `[cx, cy, w, h]` normalized to `[0, 1]` via `Sigmoid`
- Loss: MSE + Custom IoU Loss

### Task 3 — Semantic Segmentation (`segmentation.py`)
- U-Net style decoder with `ConvTranspose2d` (no bilinear interpolation)
- Skip connections: encoder feature maps concatenated channel-wise at each decoder stage
- Loss: Combined CE (α=0.5) + Soft Dice — CE provides stable per-pixel gradients, Dice directly optimizes DSC and handles class imbalance

### Task 4 — Unified Multi-Task Pipeline (`multitask.py`)
- Single `forward(x)` pass branches into all three task heads
- Returns `MultiTaskOutput` dict with keys: `classification`, `localization`, `segmentation`
- `MultiTaskLoss` combines all three losses with configurable λ weights

---

## Custom Components

### `CustomDropout` (layers.py)
Implements inverted dropout scaling **without** using `nn.Dropout` or `F.dropout`:
```python
# Training: mask ~ Bernoulli(1-p), then scale by 1/(1-p)
mask = torch.bernoulli(torch.full(x.shape, 1-p, device=x.device))
x = x * mask / (1 - p)
# Eval: identity (no scaling needed at inference)
```

### `IoULoss` (losses/iou_loss.py)
Pure PyTorch IoU loss from `[cx, cy, w, h]` inputs — no external IoU libraries:
```python
Loss = 1 - IoU = 1 - (intersection / union)
# All ops (max, min, clamp) have subgradients → backprop works correctly
```

---

## Dataset

**Oxford-IIIT Pet Dataset** — 37 pet breeds, ~7,400 images
- Classification label (0–36)
- Bounding box annotations (XML)
- Trimap segmentation masks (3 classes: foreground=0, background=1, boundary=2)

Download: https://www.robots.ox.ac.uk/~vgg/data/pets/

Expected directory layout:
data/
├── images/           # .jpg files
└── annotations/
├── trainval.txt
├── test.txt
├── xmls/         # bounding box XMLs
└── trimaps/      # segmentation masks (.png)

---

## Architecture Details

| Component | Details |
|---|---|
| Encoder | VGG11 (5 blocks, BN after every conv) |
| Classifier head | Linear(25088→4096→4096→37), CustomDropout p=0.5 |
| Localizer head | Linear(25088→1024→256→4), Sigmoid output |
| Segmenter decoder | 5× ConvTranspose2d + skip concat + double conv |
| Segmentation output | 3-class logits [B, 3, H, W] |
| Localization output | [cx, cy, w, h] normalized [0,1] |

---

## Training Setup

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Weight decay | 1e-4 |
| Image size | 224×224 |
| Normalization | ImageNet mean/std |
| Augmentation | HFlip, ColorJitter, Rotate (albumentations) |
| Seg loss α | 0.5 (equal CE + Dice) |
| Localization freeze | Encoder blocks 1–3 frozen |

---

## W&B Report

The `notebooks/da6401_wandb_report.ipynb` Kaggle notebook covers:

| Section | Content |
|---|---|
| 2.1 | BatchNorm effect on activation distributions and convergence |
| 2.2 | Dropout p=0.0 / 0.2 / 0.5 — training dynamics and generalization gap |
| 2.3 | Transfer learning showdown: Strict / Partial / Full fine-tuning on segmentation |
| 2.4 | Feature map visualization: first conv (edges) vs last conv (semantics) |
| 2.5 | Bounding box prediction table with IoU and confidence scores |
| 2.6 | Dice vs Pixel Accuracy — why Dice is superior for imbalanced segmentation |
| 2.7 | Full pipeline showcase on novel images |
| 2.8 | Meta-analysis and retrospective architectural reflection |

---

## Pretrained Checkpoints

The `MultiTaskPerceptionModel` auto-downloads checkpoints from Google Drive on first instantiation:

| Model | Drive ID |
|---|---|
| `classifier.pth` | `1QrvqfuyTOlqndMS6TdqGaJvzB5FYKRNk` |
| `localizer.pth` | `1EWb8dx2vnf_nEmD_4yvGSXuqECQ9jkCQ` |
| `unet.pth` | `120pP0rwv6Kw28qdwd6f2VgyM4s3DjwUn` |

---

## Requirements
torch>=2.0
torchvision
albumentations
numpy
matplotlib
gdown
tqdm
Pillow

Install:
```bash
pip install torch torchvision albumentations gdown tqdm matplotlib Pillow
```

---

## Quick Inference

```python
from models.multitask import MultiTaskPerceptionModel
import torch

model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
model.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(x)

print(out["classification"].shape)   # [1, 37]
print(out["localization"].shape)     # [1, 4]  — pixel-space cx,cy,w,h
print(out["segmentation"].shape)     # [1, 3, 224, 224]
```

---

## References

- Simonyan & Zisserman, *Very Deep Convolutional Networks*, ICLR 2015
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- Parkhi et al., *Cats and Dogs*, CVPR 2012
- Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR 2014

---

## Author

**Course:** DA6401 — Introduction to Deep Learning, IIT Madras  
**Assignment:** 2 — Multi-Task Visual Perception Pipeline
