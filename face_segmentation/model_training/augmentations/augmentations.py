from typing import Dict

import cv2
import albumentations as albu
import numpy as np
import torch


__all__ = ["get_transforms"]


output_format = {
    "none": lambda array: array,
    "float": lambda array: torch.FloatTensor(array),
    "long": lambda array: torch.LongTensor(array),
}

normalization = {
    "none": lambda array: array,
    "divide_by_255": lambda array: array // 255,
    "default": lambda array: albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(image=array)["image"],
    "binary": lambda array: np.array(array > 0, np.float32)
}

augmentations = {
    "weak": albu.Compose([albu.HorizontalFlip()]),
    "none": albu.Compose([]),
}

size_augmentations = {
    "none": lambda size: albu.NoOp(),
    "resize": lambda size: albu.Resize(height=size, width=size, interpolation=cv2.INTER_AREA),
    "center": lambda size: albu.CenterCrop(size, size),
    "crop_or_resize": lambda size: albu.OneOf([
        albu.RandomCrop(size, size),
        albu.Resize(height=size, width=size)
    ], p=1),
    "crop": lambda size: albu.RandomCrop(size, size),
}


def get_transforms(config: Dict):
    size = config.get("size", None)
    scope = config.get("augmentation_scope", "none")
    size_transform = config.get("size_transform", "none")

    images_normalization = config.get("images_normalization", "default")
    masks_normalization = config.get("masks_normalization", "binary")

    images_output_format_type = config.get("images_output_format_type", "float")
    masks_output_format_type = config.get("masks_output_format_type", "long")

    aug = albu.Compose(
        [augmentations[scope],
         size_augmentations[size_transform](size)]
    )

    def process(image, mask):
        if mask is not None:
            res = aug(image=image, mask=mask)
            transformed_image = output_format[images_output_format_type](
                normalization[images_normalization](res["image"])
            )
            transformed_mask = output_format[masks_output_format_type](
                normalization[masks_normalization](res["mask"])
            )
            return transformed_image, transformed_mask
        else:
            res = aug(image=image)
            transformed_image = output_format[images_output_format_type](
                normalization[images_normalization](res["image"])
            )
            return transformed_image

    return process
