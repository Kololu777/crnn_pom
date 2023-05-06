import logging
from typing import Any, Dict, List, Optional, Tuple

import lmdb
import numpy as np
import six
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from dataset.imgaug import CRNNTransform

logger = logging.getLogger(__name__)


class LmdbDataset(Dataset):
    """
    A PyTorch Dataset class for loading datasets stored in LMDB format.

    This class provides an interface for loading a dataset stored in LMDB format and returning the image and label for each sample.

    Args:
        root (str): The path to the LMDB dataset.

    Attributes:
        root (str): The path to the LMDB dataset.
        is_transform (bool): Whether to apply transformations to the image.
        dataset_index_list (list): The list of indices in the dataset.
        env (lmdb.Environment): The LMDB environment object.
        num_samples (int): The number of samples in the dataset.

    Methods:
        _get_env(): Opens the LMDB environment.
        load_lmdb_dataset(): Loads the LMDB dataset.
        get_lmdb_sample_info(): Retrieves the sample information for the given index.
        buf2PIL(): Converts a buffer to a PIL.Image object.
        buf2PIL_strict(): Strictly converts a buffer to a PIL.Image object.
        transform(): Applies transformations to the image.
        __len__(): Returns the number of samples in the dataset.
        __getitem__(): Returns the sample at the specified index.
    """

    def __init__(
        self, root, opt, transform: Optional[CRNNTransform] = CRNNTransform
    ) -> None:
        self.root = root
        self.imgW = opt["imgW"]
        self.imgH = opt["imgH"]
        self.is_keep_ratio = opt["is_keep_ratio"]
        self.is_lower_characeter = opt["is_lower_character"]
        self.transform_function = transform((self.imgH, self.imgW), self.is_keep_ratio)
        self.dataset_index_list = self.load_lmdb_dataset(root)

    def _get_env(self, root: str = "") -> lmdb.Environment:
        return lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def load_lmdb_dataset(self, root: str = "") -> List[int]:
        self.env = self._get_env(root)
        with self.env.begin(write=False) as txn:
            self.num_samples = int(txn.get("num-samples".encode()))
        # Since the key values for images and labels are 1-origin, add one.
        return [index + 1 for index in range(self.num_samples)]

    def get_lmdb_sample_info(self, index: int) -> Optional[Tuple[Image.Image, Any]]:
        with self.env.begin(write=False) as txn:
            label_key = "label-%09d".encode() % index
            label = txn.get(label_key)
            if label is None:
                return None
            label = label.decode("utf-8")
            img_key = "image-%09d".encode() % index
            img = self.buf2PIL_strict(txn, img_key)
            return img, label

    def buf2PIL(
        self, txn: lmdb.Transaction, key: bytes, type: str = "RGB"
    ) -> Image.Image:
        imgbuf = txn.get(key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert(type)
        return img

    def buf2PIL_strict(
        self, txn: lmdb.Transaction, key: bytes, type: str = "RGB"
    ) -> Image.Image:
        try:
            img = self.buf2PIL(txn, key, type)
        except IOError:
            img = Image.new(type, (self.imgW, self.imgH))
            logger.info("File Dumps. Because cannot convert buf format in PIL")
        return img

    def transform(self, img: Image.Image, label: str) -> Tuple[Tensor, str]:
        img = self.transform_function(img)
        # out_of_char = f"[^{self.opt.tokenizer}]"
        # label = re.sub(out_of_char, "", label)
        label = label.lower()
        return img, label

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        file_idx = int(self.dataset_index_list[index])
        sample_info = self.get_lmdb_sample_info(file_idx)
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        data = {"image": img, "label": label}
        if self.transform_function is not None:
            data["image"], data["label"] = self.transform(img, label)
        return data


"""
data_root = {"trainroot":"/workspace/CRNN/data/data_lmdb_release/training/MJ/MJ_train/"}

data_root2 = {"valroot":"/workspace/CRNN/data/data_lmdb_release/evaluation/IC03_860"}

opt = {"imgH":32, "imgW":100, "is_keep_ratio":False}

a = LmdbDataset(data_root["trainroot"], opt)

print(len(a))


print(a[0]['image'].size())
print(a[0]['label'])

print(a.dataset_index_list[:10])
"""

"""
print(a[0])
for i in range(10):
    print(a[i])

i = 2
image_key = f"image-{i:09d}"
print(image_key)

print(image_key.encode("utf-8").decode("utf-8"))
"""
