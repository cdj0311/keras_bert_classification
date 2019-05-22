import os
from enum import Enum


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()

xla = True
# list of int. this model has 12 layers, By default this program works on the second last layer. The last layer is too
# closed to the target functions,If you question about this argument and want to use the last hidden layer anyway, please
# feel free to set layer_indexes=[-1], so we use the second last layer
layer_indexes = [-2]
