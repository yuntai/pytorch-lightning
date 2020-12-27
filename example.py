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


model = Model()
p_model = LightningParallelModule(model)
dp_model = DataParallel(p_model, device_ids=[0, 1])