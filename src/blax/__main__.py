#!/usr/bin/env python3

# Copyright 2024 (author: lamnguyenx)


from . import __monkeypatching__
from .black import patched_main
if __name__ == "__main__":
    patched_main()
