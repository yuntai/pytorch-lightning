# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test deprecated functionality which will be removed in vX.Y.Z"""
import sys

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.plugins.legacy.ddp_plugin import DDPPlugin
from tests.base import BoringModel
from tests.deprecated_api import _soft_unimport_module


def test_v1_4_0_deprecated_imports():
    _soft_unimport_module('pytorch_lightning.utilities.argparse_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.argparse_utils import from_argparse_args  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.model_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.model_utils import is_overridden  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.warning_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.warning_utils import WarningCache  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.xla_device_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils  # noqa: F811 F401


def test_v1_4_0_deprecated_trainer_device_distrib():
    """Test that Trainer attributes works fine."""
    trainer = Trainer()
    trainer._distrib_type = None
    trainer._device_type = None

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.on_cpu = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.on_cpu

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.on_gpu = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.on_gpu

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.on_tpu = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.on_tpu
    trainer._device_type = None
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_tpu = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_tpu

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_dp = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_dp

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_ddp = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_ddp

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_ddp2 = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_ddp2

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_horovod = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_horovod


def test_v1_4_0_deprecated_metrics():
    from pytorch_lightning.metrics.functional.classification import stat_scores_multiple_classes
    with pytest.deprecated_call(match='will be removed in v1.4'):
        stat_scores_multiple_classes(pred=torch.tensor([0, 1]), target=torch.tensor([0, 1]))

    from pytorch_lightning.metrics.functional.classification import iou
    with pytest.deprecated_call(match='will be removed in v1.4'):
        iou(torch.randint(0, 2, (10, 3, 3)),
            torch.randint(0, 2, (10, 3, 3)))

    from pytorch_lightning.metrics.functional.classification import recall
    with pytest.deprecated_call(match='will be removed in v1.4'):
        recall(torch.randint(0, 2, (10, 3, 3)),
               torch.randint(0, 2, (10, 3, 3)))

    from pytorch_lightning.metrics.functional.classification import precision
    with pytest.deprecated_call(match='will be removed in v1.4'):
        precision(torch.randint(0, 2, (10, 3, 3)),
                  torch.randint(0, 2, (10, 3, 3)))

    from pytorch_lightning.metrics.functional.classification import precision_recall
    with pytest.deprecated_call(match='will be removed in v1.4'):
        precision_recall(torch.randint(0, 2, (10, 3, 3)),
                         torch.randint(0, 2, (10, 3, 3)))

    from pytorch_lightning.metrics.functional import precision

    # Testing deprecation of class_reduction arg in the *new* precision
    with pytest.deprecated_call(match='will be removed in v1.4'):
        precision(torch.randint(0, 2, (10,)),
                  torch.randint(0, 2, (10,)),
                  class_reduction='micro')

    from pytorch_lightning.metrics.functional import recall

    # Testing deprecation of class_reduction arg in the *new* recall
    with pytest.deprecated_call(match='will be removed in v1.4'):
        recall(torch.randint(0, 2, (10,)),
               torch.randint(0, 2, (10,)),
               class_reduction='micro')


class CustomDDPPlugin(DDPPlugin):

    def configure_ddp(self, model, device_ids):
        # old, deprecated implementation
        with pytest.deprecated_call(
            match='`LightningDistributedDataParallel` is deprecated since v1.2 and will be removed in v1.4.'
        ):
            model = LightningDistributedDataParallel(
                module=model,
                device_ids=device_ids,
                **self._ddp_kwargs,
            )
        return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_v1_4_0_deprecated_lightning_distributed_data_parallel(tmpdir):
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        gpus=2,
        accelerator="ddp_spawn",
        plugins=[CustomDDPPlugin()]
    )
    trainer.fit(model)
