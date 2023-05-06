from enum import Enum
from typing import Tuple, Union

import numpy as np
import torch
from dataset.imgaug import ImageConverter
from PIL import Image
from torch import Tensor


class PadDirection(Enum):
    RIGHT = "right"
    LEFT = "left"


class PAD(object):
    def __init__(self, max_size: Tuple[int, int, int], pad_direction: PadDirection = PadDirection.RIGHT):
        self.max_size = max_size
        self.pad_direction = pad_direction
        self._converter = ImageConverter()

    def __call__(self, img: Union[Image.Image, np.ndarray, Tensor]) -> Tensor:
        c, h, w = img.size()
        pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        pad_img[:, :, w] = img

        if self.max_size[2] != w and self.pad_direction == PadDirection.RIGHT:
            pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        elif self.max_size[2] != w and self.pad_direction == PadDirection.LEFT:
            pad_img[:, :, self.max_size[2] - w] = pad_img[:, :, w]
            pad_img[:, :, : self.max_size[2] - w] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return pad_img
