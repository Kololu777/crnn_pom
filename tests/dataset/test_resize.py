import numpy as np
import pytest
import torch
from PIL import Image
from src.dataset.imgaug.resize import Normalize, Resize


@pytest.fixture
def resize_transform():
    return Resize(size=(224, 224))


def test_Resize(resize_transform):
    img = Image.new(mode="RGB", size=(300, 200))
    resized_img = resize_transform(img)
    assert isinstance(resized_img, Image.Image)
    assert resized_img.size == (224, 224)


def test_Normalize():
    normalize = Normalize()
    img = Image.new(mode="RGB", size=(300, 200))
    img_np = np.array(img).shape
    normalized_img = normalize(img)
    c, h, w = normalized_img.size()
    assert isinstance(normalized_img, torch.Tensor)
    assert (c, h, w) == (img_np[2], img_np[0], img_np[1])
    assert torch.max(normalized_img) <= 1.0
    assert torch.min(normalized_img) >= -1.0
