import random


class Compose(object):
    def __init__(self, transforms) -> None:
        self._transform_register = []
        for transform in transforms:
            transform = self.transform_add_p(transform)
            self._transform_register.append(transform)

    @staticmethod
    def transform_add_p(transform_requirement):
        if isinstance(transform_requirement, tuple):
            return transform_requirement
        else:
            return (transform_requirement, 1.0)

    def __call__(self, img):
        for transform, p in self._transform_register:
            if p >= random.random():
                img = transform(img)
        return img
