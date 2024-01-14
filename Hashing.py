import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json
from sentence_transformers.util import fullname, import_from_string


class Hashing(nn.Module):
    """For Reference:
    - https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Dense.py
    - Efficient Passage Retrieval with Hashing for Open-domain Question Answering, https://arxiv.org/abs/2106.00882
    """

    def __init__(self, hashing_function):
        super(Hashing, self).__init__()
        self.hashing_function = hashing_function

    def forward(self, features: Dict[str, Tensor]):
        features.update({"sentence_embedding": self.hashing_function(features["sentence_embedding"])})
        return features

    def get_config_dict(self):
        return {"hashing_function": fullname(self.hashing_function)}

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def __repr__(self):
        return "Hashing({})".format(self.get_config_dict())

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        config["hashing_function"] = import_from_string(config["hashing_function"])()
        model = Hashing(**config)
        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model
