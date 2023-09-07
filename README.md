# Focal Tversky Loss for 3D Segmentation in PyTorch

The Focal Tversky loss is a loss function designed to handle class imbalance for segmentation tasks. It's a modification of the Tversky loss, introducing a focusing parameter, making it especially useful for medical image segmentation tasks where certain classes can be under-represented.

This repository provides an implementation of the Focal Tversky loss for 3D segmentation tasks, suitable for volumetric medical images.

## Structure of the Code

The `FocalTverskyLoss` class is a PyTorch module. The input tensors are expected to have a shape of `(B, N, H, W, L)`, where:
- `B` is the batch size
- `N` is the number of channels (e.g., 1 for binary segmentation tasks)
- `H, W, L` represent the depth, height, and width of the volumes, respectively

## Parameters

- `alpha`: Controls the magnitude of penalties for false negatives. Default is `0.7`.
- `beta`: Controls the magnitude of penalties for false positives. Default is `0.3`.
- `gamma`: Focusing parameter to weight the contribution of each voxel to the loss. Default is `0.75`.

## Usage

1. Clone this repository.
2. Import the `FocalTverskyLoss` class into your training script.
3. Instantiate the loss and use it as you would with any PyTorch loss function.

```python
from focal_tversky_loss import FocalTverskyLoss

criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
loss = criterion(predictions, ground_truth)
