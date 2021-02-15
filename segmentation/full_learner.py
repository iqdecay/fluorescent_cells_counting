from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from data_processing.datasets import MixedDataset
from models import FCRN, UNet


class FullLearner:
    def __init__(self, model: nn.Module, hyperparams: Dict[str, Any]):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.hyperparams = hyperparams

    def print_model_summary(self):
        """"Print the model's architecture summary."""
        summary(self.model, (1, 256, 256))

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int
    ):
        """"""
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.hyperparams["model_lr"],
            weight_decay=self.hyperparams["optimizer"]["weight_decay"],
        )

        for epoch in range(n_epochs):
            print(f"epoch: {epoch + 1}/{n_epochs}")
            self.model.train()

            train_outputs, train_loss = self._train_step(train_loader)
            import ipdb

            ipdb.set_trace()

    def _train_step(self, train_loader: DataLoader):
        """"""
        train_outputs, train_loss = [], []
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()
            # Perform forward propagation
            outputs, _ = self.model(inputs)
            # Compute loss and perform back propagation
            loss = nn.BCEWithLogitsLoss(pos_weight=self._calc_weights(labels))(
                outputs, labels
            )
            loss.backward()
            # Perform the weights' optimization
            self.optimizer.step()

            train_outputs.append(outputs)
            train_loss.append(train_loss)

        return train_outputs, train_loss

    def _calc_weights(self, labels):
        """"""
        pos_tensor = torch.ones_like(labels)
        for label_idx in range(0, labels.size(0)):
            pos_weight = torch.sum(labels[label_idx] == 1)
            neg_weight = torch.sum(labels[label_idx] == 0)
            ratio = float(neg_weight.item() / pos_weight.item())
            pos_tensor[label_idx] = ratio * pos_tensor[label_idx]

        return pos_tensor


if __name__ == "__main__":
    hyperparams = {
        "model_lr": 0.01,
        "optimizer": {"weight_decay": 0.0005, "momentum": 0.9},
    }

    model_name = "UNet"
    if model_name == "FCRN":
        learner = FullLearner(FCRN(), hyperparams)
        learner.print_model_summary()
    else:
        learner = FullLearner(UNet(), hyperparams)
        learner.print_model_summary()

    train_dataset = MixedDataset("./data", ["B39", "EM", "TNBC"], True)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_dataset = MixedDataset("./data", ["ssTEM"], False)
    test_loader = DataLoader(train_dataset, shuffle=True)

    learner.fit(train_loader, test_loader, 5)
