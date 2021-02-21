from collections import defaultdict
import os
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from data_processing.datasets import MixedDataset
from models import FCRN, UNet
from utils import progressbar, get_intersection_over_union


class FullLearner:
    def __init__(
        self, model: nn.Module, hyperparams: Dict[str, Any], root_dir: str
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.hyperparams = hyperparams

        self.n_early_stopping = 4
        self.early_stopping_delta = 0.8

        self.root_dir = root_dir
        self.model_name = "test"
        self.writer = SummaryWriter(f"{root_dir}/logs/{self.model_name}")

    def print_model_summary(self):
        """"Print the model's architecture summary."""
        summary(self.model, (1, 256, 256))

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int
    ):
        """Train the model and compute metrics at each epoch.

        :param n_epochs: number of epochs.
        :param train_loader: iterable train set.
        :param val_loader: iterable validation set.
        :return: history of the training (iou and loss).
        """
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.hyperparams["model_lr"],
            weight_decay=self.hyperparams["optimizer"]["weight_decay"],
        )

        history = defaultdict(list)
        best_iou = -1
        for epoch in range(n_epochs):
            print(f"epoch: {epoch + 1}/{n_epochs}")
            self.current_epoch = epoch
            self.model.train()

            # Training step
            _, _, _ = self._train_step(train_loader)

            # Evaluation step
            train_outputs, train_labels, train_loss = self.evaluate(
                train_loader,
            )
            val_outputs, val_labels, val_loss = self.evaluate(
                val_loader,
            )

            train_iou = get_intersection_over_union(
                train_outputs, train_labels, mean=True
            )
            val_iou = get_intersection_over_union(
                val_outputs, val_labels, mean=True
            )

            # Store metrics
            history["loss"].append(train_loss)
            history["iou"].append(train_iou)
            history["val_loss"].append(val_loss)
            history["val_iou"].append(val_iou)
            self._write_logs(train_loss, train_iou, val_loss, val_iou)

            # Save the best model
            if val_iou > best_iou:
                best_iou = val_iou
                self._save()

            # Print statistics at the end of each epoch
            print(
                f"loss: {train_loss:.3f}, val_loss: {val_loss:.3f}",
                f"iou: {train_iou:.3f},",
                f"val_iou: {val_iou:.3f} \n",
            )

            # Early stopping
            if (self.current_epoch >= self.n_early_stopping) and (
                history["val_iou"][-1] - self.early_stopping_delta
                > history["val_iou"][-self.n_early_stopping]
            ):
                print("Early stopping criterion reached")
                break

        history_df = pd.DataFrame(history)
        history_df.index.name = "epochs"
        self.writer.close()

    def _train_step(self, train_loader: DataLoader):
        """Train the model over the given samples.

        :param train_loader: iterable data set.
        :return: "probabilities" for each class and targets and loss.
        """
        train_outputs, train_labels, train_loss = [], [], []
        for step, (inputs, labels) in progressbar(
            enumerate(train_loader), n_steps=len(train_loader)
        ):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()
            # Perform forward propagation
            outputs, _ = self.model(inputs)
            # Compute loss and perform back propagation
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=self._calc_weights(labels)
            )
            loss = criterion(outputs, labels)
            loss.backward()
            # Perform the weights' optimization
            self.optimizer.step()

            train_outputs.append(outputs)
            train_labels.append(labels)
            train_loss.append(loss)

            self.writer.add_scalar(
                "loss/train_step",
                loss,
                self.current_epoch * len(train_loader) + step,
            )

        train_loss = torch.mean(torch.stack(train_loss)).item()
        self.writer.flush()

        return torch.cat(train_outputs), torch.cat(train_labels), train_loss

    def _calc_weights(self, labels):
        """"""
        pos_tensor = torch.ones_like(labels)
        for label_idx in range(0, labels.size(0)):
            pos_weight = torch.sum(labels[label_idx] == 1)
            neg_weight = torch.sum(labels[label_idx] == 0)
            ratio = float(neg_weight.item() / pos_weight.item())
            pos_tensor[label_idx] = ratio * pos_tensor[label_idx]

        return pos_tensor

    def evaluate(
        self, data_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Compute predictions of the model given samples.

        :param data_loader: iterable data set.
        :return: "probabilities" for each class and targets and loss.
        """
        # Set the model in evaluation mode
        self.model.eval()
        eval_outputs, eval_labels, eval_loss = [], [], []
        # Iterate over the data set without updating gradients
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.model(inputs)
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=self._calc_weights(labels)
                )
                loss = criterion(outputs, labels)

                eval_outputs.append(outputs)
                eval_labels.append(labels)
                eval_loss.append(loss)

            eval_loss = torch.mean(torch.stack(eval_loss)).item()

        return torch.cat(eval_outputs), torch.cat(eval_labels), eval_loss

    def _write_logs(
        self,
        train_loss: float,
        train_iou: float,
        val_loss: float,
        val_iou: float,
    ):
        """Write training and validation metrics in tensorboard."""
        self.writer.add_scalar(
            "loss/train_epoch",
            train_loss,
            self.current_epoch,
        )
        self.writer.add_scalar(
            "loss/val_epoch",
            val_loss,
            self.current_epoch,
        )
        self.writer.add_scalar(
            "iou/train_epoch",
            train_iou,
            self.current_epoch,
        )
        self.writer.add_scalar(
            "iou/val_epoch",
            val_iou,
            self.current_epoch,
        )

        self.writer.flush()

    def _save(self):
        """Save a model."""
        if not os.path.exists(f"{self.root_dir}/models"):
            os.makedirs(f"{self.root_dir}/models")
        saving_path = f"{self.root_dir}/models/{self.model_name}.pth"
        torch.save(self.model, saving_path)
        print("Best model saved")


if __name__ == "__main__":
    hyperparams = {
        "model_lr": 0.01,
        "optimizer": {"weight_decay": 0.0005, "momentum": 0.9},
    }

    model_name = "UNet"
    if model_name == "FCRN":
        learner = FullLearner(FCRN(), hyperparams, "./")
        learner.print_model_summary()
    else:
        learner = FullLearner(UNet(), hyperparams, "./")
        learner.print_model_summary()

    train_dataset = MixedDataset("./data", ["TNBC"], True)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_dataset = MixedDataset("./data", ["ssTEM"], False)
    test_loader = DataLoader(train_dataset, shuffle=True)

    learner.fit(train_loader, test_loader, 20)
