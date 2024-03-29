#!/usr/bin/env python3

# Copyright 2024 (author: lamnguyenx)


# technical
import typing as tp
from dataclasses import dataclass
from warnings import warn



# local
from .black.mode import Mode, Deprecated

@dataclass
class Mode__BLAX(Mode):

    string_normalization = False

    def __post_init__(self):
        self.string_normalization = False