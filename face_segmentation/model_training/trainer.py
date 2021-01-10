import os
import tqdm
import torch
import numpy as np
import torch.optim as optim

from torchvision.utils import make_grid
from face_segmentation.model_training.losses import get_loss_fn
from face_segmentation.model_training.metrics import get_metric


class Trainer:
    def __init__(self, config, train_dl, val_dl, model, summary_writer, device):
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model
        self.summary_writer = summary_writer
        self.device = device
        self.criteria = get_loss_fn(self.config["loss"])
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.metrics = {metric_name: get_metric(metric_name) for metric_name in self.config["metrics"]}
        self.class_colors = np.random.randint(0, 255, (2, 3), dtype=np.uint8)

    def _get_optimizer(self):
        optimizer_config = self.config["optimizer"]

        lr_list = optimizer_config["lr"]
        if isinstance(lr_list, list):
            param_groups = self.model.get_params_groups()
            if not len(param_groups) == len(lr_list):
                raise ValueError("Length of lr list must match number of parameter groups")
            param_lr = [{"params": group, "lr": lr_value} for group, lr_value in zip(param_groups, lr_list)]
        else:
            param_lr = [{"params": self.model.parameters(), "lr": lr_list}]

        if optimizer_config["name"] == "adam":
            return optim.Adam(param_lr,
                              weight_decay=optimizer_config.get("weight_decay", 0))

        raise TypeError(f"Unknown optimizer name: {optimizer_config['name']}")

    def _get_scheduler(self):
        scheduler_config = self.config["scheduler"]

        if scheduler_config["name"] == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                        mode="min",
                                                        factor=scheduler_config["factor"],
                                                        patience=scheduler_config["patience"])
        elif scheduler_config["name"] == "step":
            return optim.lr_scheduler.StepLR(self.optimizer,
                                             step_size=scheduler_config["step_size"],
                                             gamma=scheduler_config["gamma"])

        raise TypeError(f"Unknown scheduler type: {scheduler_config['name']}")

    def train(self):
        self.model.to(self.device)
        best_metric = -float("inf")

        for epoch in range(1, self.config["num_epochs"] + 1):
            train_loss = self._run_epoch(epoch)
            val_loss, metrics, batch_sample = self._validate()

            self._write_to_tensor_board(epoch, train_loss, val_loss, metrics, batch_sample)

            if metrics["iou"] > best_metric:
                self._save_checkpoints()
                best_metric = metrics["iou"]

            if self.config["scheduler"] == "plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            print(f"\nEpoch: {epoch}; train loss = {train_loss}; validation loss = {val_loss}")

    def _run_epoch(self, epoch):
        self.model.train()

        lr = self.optimizer.param_groups[0]["lr"]
        status_bar = tqdm.tqdm(total=len(self.train_dl))
        status_bar.set_description(f"Epoch {epoch}, lr {lr}")

        losses = []
        for x, y in self.train_dl:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            logits = self.model(x)
            loss = self.criteria(logits, y)
            loss.backward()
            self.optimizer.step()

            status_bar.update()
            status_bar.set_postfix(loss=loss.item())
            losses.append(loss.item())

        status_bar.close()
        return sum(losses) / len(losses)

    def _validate(self):
        self.model.eval()

        status_bar = tqdm.tqdm(total=len(self.val_dl))

        losses, metrics = [], {metric_name: [] for metric_name in self.metrics}
        with torch.no_grad():
            for x, y in self.val_dl:
                logits = self.model(x.to(self.device))
                loss = self.criteria(logits, y.to(self.device))
                losses.append(loss.item())

                for metric_name, metric in self.metrics.items():
                    metrics[metric_name].append(metric(logits, y))

                status_bar.update()
                status_bar.set_postfix(loss=loss.item())

        status_bar.close()
        return sum(losses) / len(losses), \
               {metric_name: sum(scores) / len(scores) for metric_name, scores in metrics.items()}, \
               {"y": y, "y_pred": logits.cpu()}

    def _save_checkpoints(self):
        torch.save(self.model.extended_state_dict(), os.path.join(self.config["log_dir"], "model.pth"))

    def _write_to_tensor_board(self, epoch, train_loss, val_loss, metrics, batch_sample):
        self.summary_writer.add_scalar(tag="Train loss", scalar_value=train_loss, global_step=epoch)
        self.summary_writer.add_scalar(tag="Validation loss", scalar_value=val_loss, global_step=epoch)

        for metric_name, metric_value in metrics.items():
            self.summary_writer.add_scalar(tag=metric_name, scalar_value=metric_value, global_step=epoch)

        images_grid = self.make_tensorboard_grid(batch_sample)
        if images_grid is not None:
            self.summary_writer.add_image("Images", images_grid, epoch)

    def make_tensorboard_grid(self, batch_sample):
        y, y_pred = batch_sample["y"], batch_sample["y_pred"]
        y_pred = (y_pred.squeeze(1) > 0).long()
        return make_grid(torch.cat([
            self.decode_segmap(y, 2),
            self.decode_segmap(y_pred.to(y.device), 2)
        ]), nrow=y.shape[0])

    def decode_segmap(self, image, nc=201):
        out = torch.empty(image.shape[0], 3, *image.shape[1:], dtype=torch.float32, device=image.device)
        for l in range(0, nc):
            idx = image == l
            out[:, 0].masked_fill_(idx, self.class_colors[l][0])
            out[:, 1].masked_fill_(idx, self.class_colors[l][1])
            out[:, 2].masked_fill_(idx, self.class_colors[l][2])

        return out / 255.0
