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
import pickle
from argparse import ArgumentParser
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.accelerators.legacy.gpu_accelerator import GPUAccelerator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerState
from tests.base import BoringDataModule, BoringModel
from tests.base.develop_utils import reset_seed


def test_can_prepare_data(tmpdir):

    dm = BoringDataModule()
    trainer = Trainer()
    trainer.datamodule = dm

    # 1 no DM
    # prepare_data_per_node = True
    # local rank = 0   (True)
    trainer.prepare_data_per_node = True
    trainer.local_rank = 0
    assert trainer.data_connector.can_prepare_data()

    # local rank = 1   (False)
    trainer.local_rank = 1
    assert not trainer.data_connector.can_prepare_data()

    # prepare_data_per_node = False (prepare across all nodes)
    # global rank = 0   (True)
    trainer.prepare_data_per_node = False
    trainer.node_rank = 0
    trainer.local_rank = 0
    assert trainer.data_connector.can_prepare_data()

    # global rank = 1   (False)
    trainer.node_rank = 1
    trainer.local_rank = 0
    assert not trainer.data_connector.can_prepare_data()
    trainer.node_rank = 0
    trainer.local_rank = 1
    assert not trainer.data_connector.can_prepare_data()

    # 2 dm
    # prepar per node = True
    # local rank = 0 (True)
    trainer.prepare_data_per_node = True
    trainer.local_rank = 0

    # is_overridden prepare data = True
    # has been called
    # False
    dm._has_prepared_data = True
    assert not trainer.data_connector.can_prepare_data()

    # has not been called
    # True
    dm._has_prepared_data = False
    assert trainer.data_connector.can_prepare_data()

    # is_overridden prepare data = False
    # True
    dm.prepare_data = None
    assert trainer.data_connector.can_prepare_data()


def test_hooks_no_recursion_error(tmpdir):
    # hooks were appended in cascade every tine a new data module was instantiated leading to a recursion error.
    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/3652
    class DummyDM(LightningDataModule):
        def setup(self, *args, **kwargs):
            pass

        def prepare_data(self, *args, **kwargs):
            pass

    for i in range(1005):
        dm = DummyDM()
        dm.setup()
        dm.prepare_data()


def test_base_datamodule(tmpdir):
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup()


def test_base_datamodule_with_verbose_setup(tmpdir):
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup('fit')
    dm.setup('test')


def test_data_hooks_called(tmpdir):
    dm = BoringDataModule()
    assert dm.has_prepared_data is False
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.prepare_data()
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.setup()
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is True


def test_data_hooks_called_verbose(tmpdir):
    dm = BoringDataModule()
    assert dm.has_prepared_data is False
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.prepare_data()
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.setup('fit')
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is False

    dm.setup('test')
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is True


def test_data_hooks_called_with_stage_kwarg(tmpdir):
    dm = BoringDataModule()
    dm.prepare_data()
    assert dm.has_prepared_data is True

    dm.setup(stage='fit')
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is False

    dm.setup(stage='test')
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is True


def test_dm_add_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = BoringDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', str(tmpdir)])
    assert args.data_dir == str(tmpdir)


def test_dm_init_from_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = BoringDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', str(tmpdir)])
    dm = BoringDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()
    assert dm.data_dir == args.data_dir == str(tmpdir)


def test_dm_pickle_after_init(tmpdir):
    dm = BoringDataModule()
    pickle.dumps(dm)


def test_train_loop_only(tmpdir):
    reset_seed()

    dm = BoringDataModule()
    model = BoringModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result
    # TODO: add end-to-end test
    # assert trainer.callback_metrics['loss'] < 0.6


def test_train_val_loop_only(tmpdir):
    reset_seed()

    dm = BoringDataModule()
    model = BoringModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result
    # TODO: add end-to-end test
    # assert trainer.callback_metrics['train_loss'] < 0.6


def test_dm_checkpoint_save(tmpdir):
    class CustomBoringModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            self.log('early_stop_on', out['x'])
            return out

    class CustomBoringDataModule(BoringDataModule):
        def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            checkpoint[self.__class__.__name__] = self.__class__.__name__

        def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            self.checkpoint_state = checkpoint.get(self.__class__.__name__)

    reset_seed()
    dm = CustomBoringDataModule()
    model = CustomBoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        weights_summary=None,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on')],
    )

    # fit model
    trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]
    checkpoint = torch.load(checkpoint_path)
    assert dm.__class__.__name__ in checkpoint
    assert checkpoint[dm.__class__.__name__] == dm.__class__.__name__


def test_test_loop_only(tmpdir):
    reset_seed()

    dm = BoringDataModule()
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.test(model, datamodule=dm)


def test_full_loop(tmpdir):
    reset_seed()

    dm = BoringDataModule()
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result

    # test
    result = trainer.test(datamodule=dm)
    # TODO: add end-to-end test
    # assert result[0]['test_acc'] > 0.8


def test_trainer_attached_to_dm(tmpdir):
    reset_seed()

    dm = BoringDataModule()
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        weights_summary=None,
        deterministic=True,
    )

    # fit model
    trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert dm.trainer is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_full_loop_single_gpu(tmpdir):
    reset_seed()

    dm = BoringDataModule()
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
        gpus=1,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result

    # test
    result = trainer.test(datamodule=dm)
    # TODO: add end-to-end test
    # assert result[0]['test_acc'] > 0.8


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_full_loop_dp(tmpdir):
    reset_seed()

    dm = BoringDataModule()
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
        accelerator='dp',
        gpus=2,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result

    # test
    result = trainer.test(datamodule=dm)
    # TODO: add end-to-end test
    # assert result[0]['test_acc'] > 0.8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_dm_transfer_batch_to_device(tmpdir):
    class CustomBatch:
        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestDM(LightningDataModule):

        hook_called = False

        def transfer_batch_to_device(self, data, device):
            self.hook_called = True
            data.samples = data.samples.to(device)
            data.targets = data.targets.to(device)
            return data

    dm = CurrentTestDM()
    model = BoringModel()

    batch = CustomBatch((torch.zeros(5, 32), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer(gpus=1)
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead
    trainer.get_model = MagicMock(return_value=model)

    model.transfer_batch_to_device = dm.transfer_batch_to_device

    trainer.accelerator_backend = GPUAccelerator(trainer)
    batch_gpu = trainer.accelerator_backend.batch_to_device(batch, torch.device('cuda:0'))
    expected = torch.device('cuda', 0)
    assert dm.hook_called
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected


def test_dm_reload_dataloaders_every_epoch(tmpdir):
    """Test datamodule, where trainer argument
    reload_dataloaders_every_epoch is set to True/False"""
    class CustomBoringDataModule(BoringDataModule):
        def __init__(self):
            super().__init__()
            self._epochs_called_for = []

        def train_dataloader(self):
            assert self.trainer.current_epoch not in self._epochs_called_for
            self._epochs_called_for.append(self.trainer.current_epoch)
            return super().train_dataloader()

    dm = CustomBoringDataModule()
    model = BoringModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=0.01,
        reload_dataloaders_every_epoch=True,
    )
    results = trainer.fit(model, dm)
    assert results
