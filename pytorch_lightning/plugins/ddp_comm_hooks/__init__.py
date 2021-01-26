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
# limitations under the License
from functools import partial

from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_7_0
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TORCH_GREATER_EQUAL_1_7_0:
    from torch.distributed.algorithms.ddp_comm_hooks import DDPCommHookType


def initialize_ddp_comm_hooks(model: LightningDistributedDataParallel, trainer, ddp_comm_hook_type):
    if not _TORCH_GREATER_EQUAL_1_7_0:
        raise MisconfigurationException(
            "Communication Hooks are introduced in PyTorch 1.7.0. Please, upgrade PyTorch to use this feature"
        )

    if ddp_comm_hook_type not in DDPCommHookType:
        raise MisconfigurationException(
            "Currently support only BuiltinCommHookType."
        )
