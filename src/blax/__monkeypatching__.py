#!/usr/bin/env python3

# Copyright 2024 (author: lamnguyenx)


import os
import typing as tp
from .booleanify import booleanify


if not booleanify(os.environ.get("JUST_BLACK", False)):

    # patching black
    from . import black
    print("targeted black:", black.__file__)

    from .super_alignments.const import STYLE_LAMNGUYENX
    black.mode.Mode.STYLE = STYLE_LAMNGUYENX

    from .lines import Line__BLAX
    black.lines.Line.get_all_properties = Line__BLAX.get_all_properties
    black.lines.Line.is_not_special = Line__BLAX.is_not_special
    black.lines.Line.__repr__ = Line__BLAX.__repr__

    from .mode import Mode__BLAX
    black.mode.Mode.string_normalization = Mode__BLAX.string_normalization
    black.mode.Mode.__post_init__ = Mode__BLAX.__post_init__

    from .lines import EmptyLineTracker__BLAX, LinesBlock__BLAX
    black.lines.EmptyLineTracker._maybe_empty_lines = (
        EmptyLineTracker__BLAX._maybe_empty_lines
    )
    black.lines.EmptyLineTracker._maybe_empty_lines_for_class_or_def = (
        EmptyLineTracker__BLAX._maybe_empty_lines_for_class_or_def
    )
    black.lines.LinesBlock.all_lines = LinesBlock__BLAX.all_lines

    from .init import _format_str_once__BLAX
    black._format_str_once = _format_str_once__BLAX


    # patching black.blib2to3
    from .black.blib2to3 import pytree as black_blib2to3_pytree
    from .blib2to3 import pytree as blax_blib2to3_pytree
    black_blib2to3_pytree.Leaf.get_str_len = blax_blib2to3_pytree.Leaf__BLAX.get_str_len
