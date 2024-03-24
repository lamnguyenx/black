#!/usr/bin/env python3

# Copyright 2024 (author: lamnguyenx)


# parent
from ..lines import Line



### FUNCTIONS
def jot_line(line: Line, name: str, step: int):

    import nice_utils
    nice_utils.jot(
        Data={
            f'{name}_str': line,
            **nice_utils.nested_unfold(
                line,
                max_depth=5,
                hiding_key_regex_str=r'.*parent.*',
            ),
        },
        name           = name,
        step           = step,
        stack_back_off = 1,
    )
