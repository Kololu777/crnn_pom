from typing import Optional, Tuple, Union

import numpy as np
import torchvision.transforms.functional as F
from dataset.imgaug import ImageConverter
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode

pair = (
    lambda x: x if isinstance(x, tuple) else (x, x)
)  # type: Callable[[Union[int, Tuple[int, int]]], Tuple[int, int]] #noqa


class PILResize:
    """A class that resizes input PIL images.

    This class resizes input images using the specified size and
    interpolation method.

    Args:
        size (int or Tuple[int, int]): The target size (width, height) for the resized image.
                                       Can be an integer or a tuple of two integers.
        interpolation (int, optional): The interpolation method used for resizing.
                                       Default is Image.BILINEAR.

    Attributes:
        size: The target size (width, height) for the resized image.
              Can be an integer or a tuple of two integers.
        interpolation: The interpolation method used for resizing.
                       Default is Image.BILINEAR.

    Examples:
        >>> resize = Resize(256)
        >>> sample_img = Image.open("sample.jpg")
        >>> resized_img = resize(sample_img)
    """

    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: int = Image.BILINEAR) -> None:
        self._interpolation = interpolation
        self._size = pair(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        """Resizes the input PIL image.

        Args:
            img: The input PIL image to be resized.

        Returns:
            A PIL image with the resized image data.
        """
        img = img.resize(self._size, self._interpolation)
        return img


class Resize:
    """A class that resizes input torch.Tensor images using torchvision.transforms.functional.resize.

    This class is a wrapper around torchvision.transforms.functional.resize and
    resizes input images using the specified size, interpolation mode, max_size, and antialiasing.

    Attributes:
        size: The target size (width, height) for the resized image.
        interpolation: The interpolation mode used for resizing. Default is InterpolationMode.BILINEAR.
        max_size: If not None, limit the image size to be no larger than this size (width, height).
                  Default is None.
        antialias: Whether to use an anti-aliasing filter when downsampling an image.
                   Default is None, and the backend decides the best approach.

    Args:
        size (Tuple[int, int]): The target size (width, height) for the resized image.
        interpolation (InterpolationMode, optional): The interpolation mode used for resizing.
                                                     Default is InterpolationMode.BILINEAR.
        max_size (Optional[Tuple[int, int]], optional): If not None, limit the image size to be no larger
                                                        than this size (width, height). Default is None.
        antialias (Optional[bool], optional): Whether to use an anti-aliasing filter when downsampling an image.
                                              Default is None, and the backend decides the best approach.

    Examples:
        >>> from torchvision.transforms import ToTensor
        >>> from PIL import Image
        >>> img = Image.open("example.jpg")
        >>> img_tensor = ToTensor()(img)
        >>> resize = Resize((128, 128))
        >>> resized_img_tensor = resize(img_tensor)
    """

    def __init__(
        self,
        size: Tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[Tuple[int, int]] = None,
        antialias: Optional[bool] = None,
    ) -> None:
        self.size = list(pair(size))
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias
        self._converter = ImageConverter()

    def __call__(self, img: Union[Image.Image, np.ndarray, Tensor]) -> Tensor:
        """Resizes the input torch.Tensor image.

        Args:
            img: The input torch.Tensor image to be resized.

        Returns:
            A torch.Tensor with the resized image data.
        """
        img = self._converter.to_torch_tensor(img)
        return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
