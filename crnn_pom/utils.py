import os
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STRLabelConverter:
    """
    This code is refered from:
    https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/utils.py

    A class to handle character encoding and decoding for Scene Text Recognition (STR) tasks.

    Attributes:
        character (str): The character set to be encoded/decoded.
        dict (dict): A dictionary representing the mapping of each character to its index.
    """

    def __init__(self, character: str):
        """
        Initialize the STRLabelConverter with the given character set.

        Args:
            character (str): The character set to be encoded/decoded.
        """
        self.character = list(character)
        self.dict = {}
        for idx, char in enumerate(list(character)):
            self.dict[char] = idx + 1

        self.character += ["[CTCblank]"] + list(character)

    def encode(self, text: Union[str, List[str]], upper_length: int = 25):
        """
        Encode the given text (or list of texts) into a LongTensor representation.

        Args:
            text (Union[str, List[str]]): The input text or list of texts to be encoded.
            upper_length (int, optional): The maximum length for the encoded text. Defaults to 25.

        Returns:
            Tuple[torch.LongTensor, torch.IntTensor]: A tuple containing the encoded text as 
            a LongTensor and the lengths as an IntTensor.
        """
        if isinstance(text, str):
            text = list(text)
        length = [len(t) for t in text]
        batch_text = torch.LongTensor(len(text), upper_length).fill_(0)
        for idx, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in list(t)]
            batch_text[idx][: len(text)] = torch.LongTensor(text)
        return batch_text.to(device), torch.IntTensor(length).to(device)

    def decode(
        self, text_index: torch.LongTensor, length: torch.IntTensor
    ) -> List[str]:
        """
        Decode the LongTensor representation of text back into its original string form.

        Args:
            text_index (torch.LongTensor): The input LongTensor containing the indices of encoded characters.
            length (torch.IntTensor): The input IntTensor containing the lengths of the texts.

        Returns:
            List[str]: A list of decoded text strings.
        """
        texts = []
        for idx, l in enumerate(length):
            t = text_index[idx, :]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = "".join(char_list)
            texts.append(text)
        return texts


def seed_initializer(seed: int):
    """
    Set the seed values for each library.

    Args:
        seed (int): The seed value to be used for initializing random seeds.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def model_save(
    model: nn.Module,
    save_dir: str,
    save_file_name: Union[int, str],
    epoch: Optional[int] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    save_file_name = os.path.splitext(str(save_file_name))[0]
    if isinstance(epoch, int):
        save_path = os.path.join(
            save_dir, f"epoch_{epoch}_step_{save_file_name}" + ".pth"
        )
    else:
        save_path = os.path.join(save_dir, f"{save_file_name}" + ".pth")
    print(save_path)
    torch.save(model.state_dict, save_path)
