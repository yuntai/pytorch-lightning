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
# Running special tests
set -e
export PL_RUNNING_SPECIAL_TESTS=1
DEFAULTS="-m coverage run --source pytorch_lightning -a -m pytest --verbose --capture=no"
python ${DEFAULTS} tests/trainer/optimization/test_manual_optimization.py::test_step_with_optimizer_closure_with_different_frequencies_ddp
python ${DEFAULTS} tests/plugins/legacy/test_rpc_plugin.py::test_rpc_function_calls_ddp
python ${DEFAULTS} tests/plugins/legacy/test_ddp_sequential_plugin.py::test_ddp_sequential_plugin_ddp_rpc_manual
python ${DEFAULTS} tests/plugins/legacy/test_ddp_sequential_plugin.py::test_ddp_sequential_plugin_ddp_rpc_manual_amp
python ${DEFAULTS} tests/plugins/legacy/test_ddp_sequential_plugin.py::test_ddp_sequential_plugin_ddp_rpc_automatic
python ${DEFAULTS} tests/utilities/test_all_gather_grad.py::test_all_gather_collection
# python ${DEFAULTS} tests/plugins/test_ddp_sequential_plugin.py::test_ddp_sequential_plugin_ddp_rpc_with_wrong_balance
python ${DEFAULTS} tests/trainer/logging_/test_train_loop_logging_1_0.py::test_logging_sync_dist_true_ddp
python ${DEFAULTS} tests/callbacks/test_pruning.py::test_pruning_callback_ddp
python ${DEFAULTS} tests/trainer/test_trainer.py::test_pytorch_profiler_trainer_ddp
python ${DEFAULTS} tests/models/test_hooks.py::test_transfer_batch_hook_ddp
