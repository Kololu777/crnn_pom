from typing import Union

import numpy as np
from dataset.imgaug import ImageConverter
from PIL import Image
from torch import Tensor


class Normalize(object):
    """A class that normalizes input PIL images.

    This class normalizes input images by converting them to tensors and
    scaling their pixel values to be between -1 and 1.

    Attributes:
        toTensor: A torchvision.transforms.ToTensor object for converting
                  PIL images to PyTorch tensors.

    Examples:
        >>> normalize = Normalize()
        >>> sample_img = Image.open("sample.jpg")
        >>> normalized_img = normalize(sample_img)
    """

    def __init__(self):
        self._converter = ImageConverter()

    def __call__(self, img: Union[Image.Image, np.ndarray, Tensor]) -> Tensor:
        """Normalizes the input PIL image.

        Args:
            img: The input PIL image to be normalized.

        Returns:
            A PyTorch tensor with the normalized image data.
        """
        img = self._converter.to_torch_tensor(img)
        return (img - 0.5) / 0.5

    def decode(self, img: Tensor) -> Tensor:
        return (img * 0.5) + 0.5
