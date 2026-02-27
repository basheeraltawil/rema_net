"""
Image / Flow Transforms
=======================
Three transform pipelines used throughout the project:

- ``rgb_transform``   : standard resize + normalise (for validation / test)
- ``flow_transform``  : resize + 2-channel normalise for optical flow
- ``train_transform`` : augmented pipeline (flips, rotations, jitter …)
"""

from torchvision import transforms


def get_transforms():
    """
    Returns:
        rgb_transform   — for validation / test RGB frames
        flow_transform  — for optical flow frames
        train_transform — augmented pipeline for training RGB frames
    """

    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize,
    ])

    flow_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
    ])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor(),
        imagenet_normalize,
    ])

    return rgb_transform, flow_transform, train_transform
