# Convolutional Recurrent Neural Network - reimplements - Pytorch
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