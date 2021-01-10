import os
import glob
import yaml
import shutil
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # read config
    with open("config.yaml") as config_file:
        config = yaml.full_load(config_file)

    # split images into train/validation parts
    img_list = glob.glob(os.path.join(config["data_path"], "images", '*.jpg'))
    train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=42)

    # copy images and masks into appropriate folders
    for img_list, folder_name in zip([train_img_list, val_img_list], ["train", "val"]):
        os.makedirs(os.path.join(config["output_path"], folder_name, "images"))
        os.makedirs(os.path.join(config["output_path"], folder_name, "masks"))

        for img_path in img_list:
            _, img_name = os.path.split(img_path)
            img_id = img_name[:-4]
            mask_path = os.path.join(config["data_path"], "masks", f"{img_id}.png")

            shutil.copy(img_path, os.path.join(config["output_path"], folder_name, "images", img_name))
            shutil.copy(mask_path, os.path.join(config["output_path"], folder_name, "masks", f"{img_id}.png"))
