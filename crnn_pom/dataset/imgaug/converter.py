from typing import Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor


class ImageConverter:
    """A class that converts an image in PIL, Torch Tensor, or NumPy array format to PIL,
    Torch Tensor, or NumPy array format.

    Example:
      >>> image_converter = ImageConverter()
      # PIL -> torch.Tensor
      >>> torch_tensor = image_converter.to_torch_tensor(pil_image)
      # PIL -> np.ndarray
      >>> numpy_array = image_converter.to_numpy_array(pil_image)
    """

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def to_torch_tensor(self, img: Union[Image.Image, np.ndarray]) -> Tensor:
        """Converts the input PIL image or NumPy array to a Torch Tensor.

        Args:
            img: The input PIL image or NumPy array to be converted.

        Returns:
            A Torch Tensor containing the image data.
        """

        if isinstance(img, np.ndarray):
            return torch.from_numpy(img.transpose((2, 0, 1)))
        elif isinstance(img, Image.Image):
            return self.to_tensor(img)
        else:
            # torch.Tensor
            return img

    def to_numpy_array(self, img: Union[Image.Image, Tensor]) -> np.ndarray:
        """Converts the input PIL image or Torch Tensor to a NumPy array.

        Args:
            img: The input PIL image or Torch Tensor to be converted.

        Returns:
            A NumPy array containing the image data.
        """
        if isinstance(img, Tensor):
            return img.numpy().transpose((1, 2, 0))
        elif isinstance(img, Image.Image):
            return np.asarray(img)
        else:
            # np.ndarray
            return img
