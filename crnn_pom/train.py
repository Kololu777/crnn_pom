from parser import opt

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from timm.optim.optim_factory import create_optimizer_v2
from tqdm import tqdm

from utils import (STRLabelConverter, model_save,
                                        seed_initializer)
from dataset.lmdb_dataset import LmdbDataset
from model.crnn import CRNN
from model.recipe import crnn_cnn, crnn_lstm

def train():
    accelerator = Accelerator()
    device = accelerator.device
    seed_initializer(opt.seed)

    model = CRNN(
        cnn_recipe=crnn_cnn,
        rnn_recipe=crnn_lstm,
        input_channels=3,
        out_features=len(opt.character) + 1,
    )

    images_opt = {
        "imgH": 32,
        "imgW": 100,
        "is_keep_ratio": False,
        "is_lower_character": True,
    }
    dataset = train_dataset = LmdbDataset(root=opt.trainroot, opt=images_opt)
    data = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.CTCLoss()
    optimizer = create_optimizer_v2(model.parameters(), opt=opt.optimizer, lr=opt.lr)

    model, optimizer, data = accelerator.prepare(model, optimizer, data)

    model.train()

    str_label_converter = STRLabelConverter(opt.character)


    for epoch in range(opt.epoch):
        for idx, data in enumerate(tqdm(data)):
            images = data["image"].to(device)
            targets_true, length_true = str_label_converter.encode(data["label"])
            targets_true = targets_true.to(device)
            length_true = length_true.to(device)

            optimizer.zero_grad()

            output_pred = model(images)
            length_pred = torch.IntTensor([output_pred.size(0)] * opt.batch_size)
            loss = criterion(output_pred, targets_true, length_pred, length_true)
            accelerator.backward(loss)
            optimizer.step()

            if idx % 2000 == 0:
                print(f"loss:{loss}")
                print(data["label"])
                _, output_pred_max = output_pred.max(2)
                output_pred_max = output_pred_max.transpose(1, 0).contiguous()
                print(str_label_converter.decode(output_pred_max, length_true))
                model_save(model, save_dir=opt.save_dir, save_file_name=idx, epoch=epoch)
        model_save.save(model, save_dir=opt.save_dir, save_file_name="last")
        
if __name__ == "__main__":
    train()
