import torch.nn as nn
from torchsummary import summary

from models import FCRN, UNet


class MetaLearner:
    def __init__(self, model: nn.Module):
        self.model = model

    def print_model_summary(self):
        """"Print the model's architecture summary."""
        summary(self.model, (1, 256, 256))


if __name__ == "__main__":
    learner = MetaLearner(FCRN())
    learner.print_model_summary()
    print("\n")
    learner = MetaLearner(UNet())
    learner.print_model_summary()
