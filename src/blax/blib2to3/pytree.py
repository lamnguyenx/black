#!/usr/bin/env python3

# Copyright 2024 (author: lamnguyenx)


# technical
import typing as tp


# local
from ..black.blib2to3.pytree import Leaf



### CLASSES
class Leaf__BLAX(Leaf):

    code_level : tp.Optional[int] = None

    def get_str_len(self) -> int:
        return len(self.prefix) + len(self.value)
