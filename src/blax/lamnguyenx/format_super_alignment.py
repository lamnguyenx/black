#!/usr/bin/env python3

# Copyright 2023 (author: lamnguyenx)


# technical
from typing_extensions import Self
import dataclasses as dcl
import typing as tp


# parent
from ..lines import LinesBlock, Line
from ..mode import Mode
from ..blib2to3.pytree import Leaf
from ..blib2to3.pgen2 import token


# nice_utils
# import nice_utils as hnu; logger = hnu.getLogger(__name__)


### CLASSES
@dcl.dataclass
class SimpleLocator:

    block_index: int
    block_line_index: int
    leaf_index: int
    code_level: int
    str_distance: int



@dcl.dataclass
class ComplexLocator:


    kind: tp.Literal['colon', 'equal', 'colon_equal']
    complex_code_level: int
    Colon_Locator: tp.Optional[SimpleLocator]
    Equal_Locator: tp.Optional[SimpleLocator]
    lineno: int
    line_str: str



    def update_str_distance(self, Inp_Leaves: tp.List[Leaf]) -> Self:

        if self.Colon_Locator is not None:
            self.Colon_Locator.str_distance = sum(
                [leaf.get_str_len() for leaf in Inp_Leaves[: self.Colon_Locator.leaf_index]]
            )

        if self.Equal_Locator is not None:
            self.Equal_Locator.str_distance = sum(
                [leaf.get_str_len() for leaf in Inp_Leaves[: self.Equal_Locator.leaf_index]]
            )

        return self



### FUNCTIONS
def count_leading_spaces(inp_str: str) -> int:
    return len(inp_str) - len(inp_str.lstrip(' '))

def count_ending_spaces(inp_str: str) -> int:
    return len(inp_str) - len(inp_str.rstrip(' '))



def format_super_alignment(Inp_Lines_Blocks: tp.List[LinesBlock], mode: Mode):

    '@lamnguyenx'

    # pre-visit (admittedly ugly) for super-alignment
    Complex_Locators: tp.List[ComplexLocator] = []
    for b, Curr_Lines_Block in enumerate(Inp_Lines_Blocks):
        Curr_First_Line    = Curr_Lines_Block.Content_Lines[0]
        cur_is_def         = Curr_First_Line.is_def
        cur_is_not_special = Curr_First_Line.is_not_special

        if cur_is_not_special or cur_is_def:

            if cur_is_def:
                starting = 1
            else:
                starting = 0

            for i in range(len(Curr_Lines_Block.Content_Lines))[starting:]:
                line = Curr_Lines_Block.Content_Lines[i]

                Colon_Locator = None
                for j, leaf in enumerate(line.leaves):
                    if leaf.type == token.COLON:
                        Colon_Locator = SimpleLocator(
                            block_index      = b,
                            block_line_index = i,
                            leaf_index       = j,
                            code_level       = leaf.code_level,
                            str_distance=sum([leaf.get_str_len() for leaf in line.leaves[:j]]),
                        )
                        break

                Equal_Locator = None
                if not cur_is_def:
                    for j, leaf in enumerate(line.leaves):
                        if leaf.type == token.EQUAL:
                            Equal_Locator = SimpleLocator(
                                block_index      = b,
                                block_line_index = i,
                                leaf_index       = j,
                                code_level       = leaf.code_level,
                                str_distance=sum([leaf.get_str_len() for leaf in line.leaves[:j]]),
                            )
                            break

                if all([Colon_Locator, Equal_Locator]):

                    if cur_is_not_special:
                        if Colon_Locator.leaf_index < Equal_Locator.leaf_index:

                            if Colon_Locator.code_level == Equal_Locator.code_level:
                                pass

                            elif Colon_Locator.code_level < Equal_Locator.code_level:
                                Equal_Locator = None

                            else:
                                Colon_Locator = None
                                Equal_Locator = None

                        else:
                            Colon_Locator = None
                            Equal_Locator = None

                    # if cur_is_def:
                    #     if not (
                    #         Colon_Locator.code_level == Curr_First_Line.leaves[0].code_level + 2
                    #         and Colon_Locator.code_level == Equal_Locator.code_level + 1
                    #         and Colon_Locator.leaf_index < Equal_Locator.leaf_index
                    #     ):
                    #         Colon_Locator = None
                    #         Equal_Locator = None

                if cur_is_def and Colon_Locator is None and Equal_Locator is not None:
                    Equal_Locator = None

                if any([Colon_Locator, Equal_Locator]):

                    code_level_List = []

                    complex_code_level = min(
                        [L.code_level for L in [Colon_Locator, Equal_Locator] if L is not None]
                    )

                    if all([Colon_Locator, Equal_Locator]):
                        kind = 'colon_equal'
                    elif Colon_Locator is not None:
                        kind = 'colon'
                    elif Equal_Locator is not None:
                        kind = 'equal'
                    else:
                        raise NotImplementedError('one of locators must be not None')

                    Complex_Locator = ComplexLocator(
                        line_str           = str(line),
                        complex_code_level = complex_code_level,
                        lineno             = leaf.lineno,
                        Colon_Locator      = Colon_Locator,
                        Equal_Locator      = Equal_Locator,
                        kind               = kind,
                    )

                    Complex_Locators.append(Complex_Locator)

    # logger.green('Complex_Locators:', Complex_Locators)

    format__Complex_Locators(
        Complex_Locators = Complex_Locators,
        Inp_Lines_Blocks = Inp_Lines_Blocks,
        mode             = mode,
    )



def format__Complex_Locators(
    Complex_Locators : tp.List[ComplexLocator],
    Inp_Lines_Blocks : tp.List[LinesBlock],
    mode             : Mode,
):


    # grouping
    if len(Complex_Locators) == 0:
        return

    group = 0
    Complex_Locator_Grouped: tp.Dict[int, tp.List[ComplexLocator]] = {}
    Complex_Locator_Grouped[group] = [Complex_Locators[0]]

    for Curr_CL in Complex_Locators[1:]:
        Prev_CL = Complex_Locator_Grouped[group][-1]

        if not (
            Curr_CL.kind == Prev_CL.kind
            and Curr_CL.lineno == Prev_CL.lineno + 1
            and Curr_CL.complex_code_level == Prev_CL.complex_code_level
        ):
            group += 1
            Complex_Locator_Grouped[group] = []

        Complex_Locator_Grouped[group].append(Curr_CL)

    for group in [g for g, L in Complex_Locator_Grouped.items() if len(L) <= 2]:
        Complex_Locator_Grouped.pop(group)

    # logger.green('Complex_Locator_Grouped:', Complex_Locator_Grouped)

    # adding spaces
    for group, Complex_Locators in Complex_Locator_Grouped.items():

        # colon
        if Complex_Locators[0].kind in ('colon', 'colon_equal'):

            d_max = max(
                [Complex_Locator.Colon_Locator.str_distance for Complex_Locator in Complex_Locators]
            )

            if len(Complex_Locators) == 2:
                d_min = min([
                    Complex_Locator.Colon_Locator.str_distance
                    for Complex_Locator in Complex_Locators
                ])

                if d_max / d_min > 2.5:
                    continue

            if d_max <= mode.STYLE.colon_distance_threshold:
                for Complex_Locator in Complex_Locators:

                    b = Complex_Locator.Colon_Locator.block_index
                    i = Complex_Locator.Colon_Locator.block_line_index
                    j = Complex_Locator.Colon_Locator.leaf_index
                    d = Complex_Locator.Colon_Locator.str_distance

                    Cur_Leaves = Inp_Lines_Blocks[b].Content_Lines[i].leaves
                    Cur_Leaves[j].prefix += ' ' * (
                        d_max - d - (count_leading_spaces(Cur_Leaves[j].prefix)) + 1
                    )

                    if j + 1 <= len(Cur_Leaves) - 1:
                        if not (
                            Cur_Leaves[j + 1].prefix.startswith(' ')
                            or Cur_Leaves[j + 1].value.startswith(' ')
                        ):
                            Cur_Leaves[j + 1].prefix = ' ' + Cur_Leaves[j + 1].prefix

                    Complex_Locator.update_str_distance(Inp_Leaves=Cur_Leaves)

        # equal
        if Complex_Locators[0].kind == 'equal':

            d_max = max(
                [Complex_Locator.Equal_Locator.str_distance for Complex_Locator in Complex_Locators]
            )

            if len(Complex_Locators) == 2:
                d_min = min([
                    Complex_Locator.Equal_Locator.str_distance
                    for Complex_Locator in Complex_Locators
                ])

                if d_max / d_min > 2.5:
                    continue

            if d_max <= mode.STYLE.equal_distance_threshold:
                for Complex_Locator in Complex_Locators:

                    b = Complex_Locator.Equal_Locator.block_index
                    i = Complex_Locator.Equal_Locator.block_line_index
                    j = Complex_Locator.Equal_Locator.leaf_index
                    d = Complex_Locator.Equal_Locator.str_distance

                    Cur_Leaves = Inp_Lines_Blocks[b].Content_Lines[i].leaves
                    Cur_Leaves[j].prefix += ' ' * (
                        d_max - d - (count_leading_spaces(Cur_Leaves[j].prefix)) + 1
                    )

                    if j + 1 <= len(Cur_Leaves) - 1:
                        if not (
                            Cur_Leaves[j + 1].prefix.startswith(' ')
                            or Cur_Leaves[j + 1].value.startswith(' ')
                        ):
                            Cur_Leaves[j + 1].prefix = ' ' + Cur_Leaves[j + 1].prefix

        elif Complex_Locators[0].kind == 'colon_equal':

            d_max__colon = max(
                [Complex_Locator.Colon_Locator.str_distance for Complex_Locator in Complex_Locators]
            )

            d_max__equal = max(
                [Complex_Locator.Equal_Locator.str_distance for Complex_Locator in Complex_Locators]
            )

            if len(Complex_Locators) == 2:
                d_min__colon = min([
                    Complex_Locator.Colon_Locator.str_distance
                    for Complex_Locator in Complex_Locators
                ])

                if d_max__colon / d_min__colon > 2.5:
                    continue

                d_min__equal = min([
                    Complex_Locator.Equal_Locator.str_distance
                    for Complex_Locator in Complex_Locators
                ])

                if d_max__equal / d_min__equal > 2.5:
                    continue

            if (d_max__equal - d_max__colon) <= mode.STYLE.equal_distance_threshold:
                # logger.red('Complex_Locators:', Complex_Locators)
                for Complex_Locator in Complex_Locators:

                    b = Complex_Locator.Equal_Locator.block_index
                    i = Complex_Locator.Equal_Locator.block_line_index
                    j = Complex_Locator.Equal_Locator.leaf_index
                    d = Complex_Locator.Equal_Locator.str_distance

                    Cur_Leaves = Inp_Lines_Blocks[b].Content_Lines[i].leaves
                    Cur_Leaves[j].prefix += ' ' * (
                        d_max__equal - d - (count_leading_spaces(Cur_Leaves[j].prefix)) + 1
                    )

                    if j + 1 <= len(Cur_Leaves) - 1:
                        if not (
                            Cur_Leaves[j + 1].prefix.startswith(' ')
                            or Cur_Leaves[j + 1].value.startswith(' ')
                        ):
                            Cur_Leaves[j + 1].prefix = ' ' + Cur_Leaves[j + 1].prefix
