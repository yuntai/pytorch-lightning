import copy
import errno
import os
import os.path as osp
import ssl
from io import BytesIO
from typing import List
from urllib.request import urlopen, urlretrieve
from zipfile import ZipFile

import torch
from six.moves import urllib

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def to_list(x):
    if not isinstance(x, (tuple, list)):
        x = [x]
    return x


def files_exist(files):
    return len(files) != 0 and all(osp.exists(f) for f in files)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def download_data(url, path="data/"):
    """
    Downloads data automatically from the given url to the path. Defaults to data/ for the path.
    Automatically handles:
        - .csv
        - .zip
    Args:
        url:
        path:
    Returns:
    """

    if ".zip" in url:
        download_zip_data(url, path)

    else:
        download_generic_data(url, path)


def download_zip_data(url, path="data/"):
    """
    Example::
        from pl_flash.data import download_zip_data
        # download titanic data
        download_zip_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "data/")
    Args:
        url: must end with .zip
        path: path to download to
    Returns:
    """
    with urlopen(url) as resp:
        with ZipFile(BytesIO(resp.read())) as file:
            file.extractall(path)


def download_generic_data(url, path="data/"):
    """
    Downloads an arbitrary file.
    Example::
        from pl_flash.data import download_csv_data
        # download titanic data
        download_csv_data("https://pl-flash-data.s3.amazonaws.com/titanic.csv", "titanic.csv")
    Args:
        url: must end with .csv
        path: path to download to (include the file name)
    Returns:
    """
    urlretrieve(url, path)


class DataType(LightningEnum):
    REQUEST_DATA = 'request_data'
    RAW_DATA = 'raw_data'
    PRE_TRANSFORM_DATA = 'pre_transform_data'
    AFTER_PRE_TRANSFORM_DATA = 'after_pre_transform_data'


class LightningDataset(torch.utils.data.Dataset):
    r"""Dataset base class for creating graph datasets.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html>`__ for the accompanying tutorial.
    Args:
        root (string, optional):
        transform (callable, optional):
        pre_transform:
        pre_filter (callable, optional): A funciton
    """
    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

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

    def __init__(self,
                 name=None,
                 root=None,
                 raw_dir=None,
                 processed_dir=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(LightningDataset, self).__init__()

        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))

        if isinstance(raw_dir, str):
            raw_dir = osp.expanduser(osp.normpath(raw_dir))

        if isinstance(processed_dir, str):
            processed_dir = osp.expanduser(osp.normpath(processed_dir))

        self.name = name
        self.root = root
        self._raw_dir = raw_dir
        self._processed_dir = processed_dir
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.__indices__ = None

        # switch between modes
        self._prediction_mode = False

        # data holder for inference injection
        self._request_data = []
        self._raw_data = []
        self._pre_transform_data = []
        self._after_pre_transform_data = []

        if 'download' in self.__class__.__dict__.keys():
            self._download()

        if 'process' in self.__class__.__dict__.keys():
            self._process()

    @property
    def _prediction_data(self):
        return self._request_data + self._raw_data + self._pre_transform_data + self._after_pre_transform_data

    def _get_prediction_data(self, index):
        if index > len(self._prediction_data):
            raise MisconfigurationException(
                "The index is out of bound for prediction data."
            )

        if index <= self._prediction_data_size(0):
            return self._request_data[index], DataType.REQUEST_DATA

        if self._prediction_data_size(0) < index and index <= self._prediction_data_size(1):
            index_ = index - self._prediction_data_size(0)
            return self._raw_data[index_], DataType.RAW_DATA

        if self._prediction_data_size(1) < index and index <= self._prediction_data_size(2):
            index_ = index - self._prediction_data_size(1)
            return self._pre_transform_data[index_], DataType.PRE_TRANSFORM_DATA

        if self._prediction_data_size(2) < index and index <= self._prediction_data_size(3):
            index_ = index - self._prediction_data_size(2)
            return self._after_pre_transform_data[index_], DataType.AFTER_PRE_TRANSFORM_DATA

    def _process_prediction_dataset(self, data, data_type):

        if data_type == DataType.REQUEST_DATA:
            data = self.request_data_to_raw(data)
            data_type = DataType.PRE_TRANSFORM_DATA

        if data_type == DataType.RAW_DATA:
            data_type = DataType.PRE_TRANSFORM_DATA

        if  data_type == DataType.PRE_TRANSFORM_DATA:
            data = self.pre_transform(data) if self.pre_transform is not None else data
            data_type = DataType.AFTER_PRE_TRANSFORM_DATA

        if data_type == DataType.AFTER_PRE_TRANSFORM_DATA:
            return data

    @property
    def raw_dir(self):
        if self._raw_dir is not None:
            return self._raw_dir
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        if self._processed_dir is not None:
            return self._processed_dir
        return osp.join(self.root, 'processed')

    @property
    def __len__(self) -> int:
        if self._prediction_mode:
            return len(self._prediction_data)
        else:
            return self.dataset_len()

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    def get_raw_data_metadata(self, index):
        return {}

    def get_raw_data(self, index, **kwargs):
        return None

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_transform):
            log.info(
                'The `pre_transform` argument differs from the one used in '
                'the pre-processed version of this dataset. If you really '
                'want to make use of another pre-processing technique, make '
                'sure to delete `{}` first.'.format(self.processed_dir))
        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_filter):
            log.info(
                'The `pre_filter` argument differs from the one used in the '
                'pre-processed version of this dataset. If you really want to '
                'make use of another pre-fitering technique, make sure to '
                'delete `{}` first.'.format(self.processed_dir))

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        print('Processing...')

        makedirs(self.processed_dir)

        datas = []

        for data_idx in range(len(self)):
            if self._prediction_mode:
                data, data_type = self._get_prediction_data(data_idx)
            else:
                metadata = self.get_raw_data_metadata(data_idx)
                data = self.get_raw_data(data_idx, **metadata)
                data_type = DataType.RAW_DATA

            data = self._process_prediction_dataset(data, data_type)

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(__repr__(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(__repr__(self.pre_filter), path)

        print('Done!')

    def add_request_data(self, request_data: List):
        self._request_data.extend(request_data)

    def add_raw_data(self, raw_data: List):
        self._raw_data.extend(raw_data)

    def add_pre_transform_data(self, pre_transform_data: List):
        self._pre_transform_data.extend(pre_transform_data)

    def indices(self):
        if self.__indices__ is not None:
            return self.__indices__
        else:
            return range(len(self))

    def __len__(self):
        r"""The number of examples in the dataset."""
        if self.__indices__ is not None:
            return len(self.__indices__)
        return self.len()

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, int):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)

    def index_select(self, idx):
        indices = self.indices()
        if isinstance(idx, slice):
            indices = indices[idx]
        elif torch.is_tensor(idx):
            if idx.dtype == torch.long:
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                return self.index_select(idx.tolist())
            elif idx.dtype == torch.bool or idx.dtype == torch.uint8:
                return self.index_select(
                    idx.nonzero(as_tuple=False).flatten().tolist())
        elif isinstance(idx, list) or isinstance(idx, tuple):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                'Only integers, slices (`:`), list, tuples, and long or bool '
                'tensors are valid indices (got {}).'.format(
                    type(idx).__name__))

        dataset = copy.copy(self)
        dataset.__indices__ = indices
        return dataset

    def shuffle(self, return_perm=False):
        r"""Randomly shuffles the examples in the dataset.
        Args:
            return_perm (bool, optional): If set to :obj:`True`, will
                additionally return the random permutation used to shuffle the
                dataset. (default: :obj:`False`)
        """
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def _prediction_data_size(self, index):
        pred_len_at_index = 0
        if index <= 0:
            pred_len_at_index += len(self._request_data)

        if index <= 1:
            pred_len_at_index += len(self._raw_data)

        if index <= 2:
            pred_len_at_index += len(self._request_data)

        if index <= 3:
            pred_len_at_index += len(self._after_pre_transform_data)

        return pred_len_at_index

    def __repr__(self):  # pragma: no cover
        return f'{self.__class__.__name__}({len(self)})'
