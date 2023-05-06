import math
from typing import Callable, Tuple, Union # noqa F401

from dataset.imgaug import PAD, Compose, Normalize, Resize

pair = (
    lambda x: x if isinstance(x, tuple) else (x, x)
)  # type: Callable[[Union[int, Tuple[int, int]]], Tuple[int, int]]


class CRNNTransform(object):
    # https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/dataset.py noqa

    def __init__(
        self, size: Union[int, Tuple[int, int]], keep_ratio_with_pad: bool  # (H, W)
    ):
        self._size = pair(size)
        self._imgH = self._size[0]
        self._imgW = self._size[1]
        self._keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, image):
        if self._keep_ratio_with_pad:
            input_channel = 3 if image.mode == "RGB" else 1
            w, h = image.size
            ratio = w / float(h)
            # Clip sup(self._imgW)
            resized_w = min(math.ceil(self._imgH * ratio), self._imgW)
            transform = Compose(
                transform=[
                    Resize((self._imgH, resized_w)),
                    Normalize(),
                    PAD(input_channel, self.imgH, self.imgW),
                ]
            )
        else:
            transform = Compose(
                transforms=[Resize((self._imgH, self._imgW)), Normalize()]
            )
            img_tensors = transform(image)
        return img_tensors
