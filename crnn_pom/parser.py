import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--trainroot",
    type=str,
    default="/workspace/CRNN/data/data_lmdb_release/training/MJ/MJ_train/",
)
parser.add_argument(
    "--valroot",
    type=str,
    default="/workspace/CRNN/data/data_lmdb_release/training/MJ/MJ_valid/",
)
parser.add_argument(
    "--character", type=str, default="0123456789abcdefghijklmnopqrstuvwxyz"
)
parser.add_argument("--optimizer", type=str, default="adadelta")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument(
    "--save_dir", type=str, default="/workspace/CRNN_reimplement/save_model_pth"
)
opt = parser.parse_args()
