import os
import yaml
import random
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from face_segmentation.model_training.datasets import PeopleDataset
from face_segmentation.model_training.augmentations import get_transforms
from face_segmentation.model_training.models import get_network
from face_segmentation.model_training.trainer import Trainer


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # read config
    with open("config.yaml") as config_file:
        config = yaml.full_load(config_file)

    # create folder for logs
    experiment_name = f"{config['model']['arch']}-{config['model']['encoder']}-" \
                      f"{config['model'].get('encoder_weights', 'none')}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    config["log_dir"] = os.path.join(config["log_dir"], experiment_name)
    os.makedirs(config["log_dir"])

    # create summary-writer
    summary_writer = SummaryWriter(config["log_dir"])

    # copy config into logs dir
    with open(os.path.join(config["log_dir"], "config.yaml"), "w") as config_copy:
        yaml.dump(config, config_copy)

    # create data transforms
    train_transforms = get_transforms(config["train"]["transform"])
    val_transforms = get_transforms(config["val"]["transform"])

    # create data-loaders
    train_ds = PeopleDataset(config["train"]["path"], train_transforms)
    val_ds = PeopleDataset(config["val"]["path"], train_transforms)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"])

    # create model
    model = get_network(config["model"])

    # configure device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # train model
    trainer = Trainer(config, train_dl, val_dl, model, summary_writer, device)
    trainer.train()
