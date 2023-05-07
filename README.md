# Convolutional Recurrent Neural Network - reimplements - Pytorch
[![Python 3.7+](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![isort](https://img.shields.io/badge/code%20style-isort-f1c40f.svg)](https://pycqa.github.io/isort/) [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![flake8](https://img.shields.io/badge/code%20style-flake8-5ed9c7.svg)](https://flake8.pycqa.org/)


This softwere is a reimplements of Convolutional Recurrent Neural Network(CRNN):[arxiv](https://arxiv.org/abs/1507.05717).
The model was implemented using PyTorch and einops. The training is a concise implementation using timm and Hugging Face's Accelerate.

# CRNN

> [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717).


# Usage

Train code sample
```
% python train.py -d ./data_lmdb_release/training/MJ/MJ_train/data.mdb
```

Inference code sample[Todo]
```
% python predict.py -i sample.png
->available
```

# TODO
- [ ] test code 
- [ ] class document 
- [ ] predict code 
- [ ] pypl

# Acknowledgements
This softwere was mostly base on https://github.com/meijieru/crnn.pytorch.