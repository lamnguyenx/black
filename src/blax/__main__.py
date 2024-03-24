#!/usr/bin/env python3

# Copyright 2024 (author: lamnguyenx)


import black
from .super_alignments.const import STYLE_LAMNGUYENX
black.mode.Mode.STYLE = STYLE_LAMNGUYENX

from black import patched_main
if __name__ == "__main__":
    patched_main()