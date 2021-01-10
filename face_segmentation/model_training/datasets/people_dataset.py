import os
import glob
import torch
import cv2


__all__ = ["PeopleDataset"]


class PeopleDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, include_mask=True):
        self.transform = transform
        self.include_mask = include_mask
        self.x_path = os.path.join(path, "images")
        self.y_path = os.path.join(path, "masks")
        self.img_list = glob.glob(os.path.join(self.x_path, "*.jpg"))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img_path = self.img_list[i]
        _, img_name = os.path.split(img_path)
        img_id = img_name[:-4]
        mask_path = os.path.join(self.y_path, f"{img_id}.png")

        x = cv2.imread(img_path)

        if self.include_mask:
            y = cv2.imread(mask_path, 0)
            x, y = self.transform(image=x, mask=y)
            return x.permute(2, 0, 1), y
        else:
            x = self.transform(image=x, mask=None)
            return x.permute(2, 0, 1)
