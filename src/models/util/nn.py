import functools
from typing import List

from torch import nn


def tie_layers(tie1: nn.Module, tie2_list: List[nn.Module], layers_names_to_tie: List[str]):
    def rsetattr(obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(rgetattr(obj, pre) if pre else obj, post, val)

    def rgetattr(obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split('.'))

    for module in tie2_list:
        for tie_name in layers_names_to_tie:
            rsetattr(module, tie_name, rgetattr(tie1, tie_name))