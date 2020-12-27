from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import DataParallel

from pytorch_lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningParallelModule


class Model(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(10, 10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print("running train step")
        out = self(batch)
        return OrderedDict({"loss": out.sum()})

    def test_step(self, batch, batch_idx):
        print("running test step")
        out = self(batch)
        return OrderedDict({"loss": out.sum()})

    def validation_step(self, batch, batch_idx):
        print("running val step")
        out = self(batch)
        return OrderedDict({"loss": out.sum()})


if __name__ == "__main__":
    model = Model()

    p_model = LightningParallelModule(model)
    p_model.to(torch.device("cuda", 0))
    dp_model = DataParallel(p_model, device_ids=[0, 1])

    model.training = True
    model.testing = False
    batch = torch.rand(5, 10, device=torch.device("cuda", 0))
    loss = dp_model(batch, 0)
    print(loss)

    model.training = False
    model.testing = True
    batch = torch.rand(5, 10, device=torch.device("cuda", 0))
    loss = dp_model(batch, 0)
    print(loss)

    model.training = False
    model.testing = False
    batch = torch.rand(5, 10, device=torch.device("cuda", 0))
    loss = dp_model(batch, 0)
    print(loss)

