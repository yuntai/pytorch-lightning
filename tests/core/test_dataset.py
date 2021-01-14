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
import pandas as pd

from pytorch_lightning.core.dataset import download_url, LightningDataset


def test_lightning_dataset(tmpdir):

    class Planetoid(LightningDataset):

        url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

        def __init__(self,
                    name=None,
                    root=None,
                    raw_dir=None,
                    processed_dir=None,
                    transform=None,
                    pre_transform=None,
                    pre_filter=None):
            super(Planetoid, self).__init__(
                name=name, root=root, raw_dir=raw_dir, processed_dir=processed_dir, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        @property
        def raw_file_names(self):
            r"""The name of the files to find in the :obj:`self.raw_dir` folder in
            order to skip the download."""
            names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
            return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

        @property
        def processed_file_names(self):
            r"""The name of the files to find in the :obj:`self.processed_dir`
            folder in order to skip the processing."""
            return ['training.pt', 'test.pt']

        def download(self):
            r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""

        def process(self):
            r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
            raise NotImplementedError

        def dataset_len(self):
            raise NotImplementedError

        def get(self, idx):
            r"""Gets the data object at index :obj:`idx`."""
            raise NotImplementedError

        def get_raw_data_metadata(self, index):
            r"""Gets the raw metadata to get raw data at index :obj:`idx`."""
            raise NotImplementedError

        def get_raw_data(self, index, **kwargs):
            r"""Gets the raw data at index :obj:`idx`."""
            raise NotImplementedError

        def request_data_to_raw(self, request_data):
            r"""Process a request data to raw """
            raise NotImplementedError

        def collate_fn(self, data_list):
            r"""Collate function to create batch of data """
            raise NotImplementedError

    dataset = Planetoid(root=tmpdir, raw_dir='/data/raw')
