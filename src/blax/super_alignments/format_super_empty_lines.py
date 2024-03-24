#!/usr/bin/env python3

# Copyright 2023 (author: lamnguyenx)


# technical
import dataclasses as dc
import typing as tp
from ..lines import LinesBlock, Line
from ..mode import Mode


# # nice_utils
# import nice_utils
# logger = nice_utils.getLogger(__name__)


### CLASSES
@dc.dataclass
class SomeGroup:

    Parent_Lines_Block: LinesBlock = dc.field(repr=False)
    Child_Lines_Blocks: tp.List[LinesBlock] = dc.field(repr=False)

    depth: int
    unpadded_height: int
    maybe_str: str = ''



### FUNCTIONS
def format_super_empty_lines(
    Inp_Lines_Blocks : tp.List[LinesBlock],
    parent           : tp.Literal['def', 'class'],
    mode             : Mode,
):

    if len(Inp_Lines_Blocks) == 0:
        return

    assert parent in ['def', 'class']

    # consider all defs/classes
    group = -1
    Group_Dict: tp.Dict[int, SomeGroup] = {}

    for Curr_Lines_Block in Inp_Lines_Blocks:
        Curr_First_Line = Curr_Lines_Block.Content_Lines[0]

        if getattr(Curr_First_Line, f'is_{parent}'):
            group += 1
            Group_Dict[group] = SomeGroup(
                Parent_Lines_Block = Curr_Lines_Block,
                Child_Lines_Blocks = list(),
                depth              = Curr_First_Line.depth,
                unpadded_height    = len(Curr_Lines_Block.Content_Lines),
                maybe_str          = ''.join(Curr_Lines_Block.all_lines()).strip(),
            )

        elif (group >= 0) and (group in Group_Dict):
            if Curr_First_Line.depth > Group_Dict[group].depth:
                Group_Dict[group].unpadded_height += len(Curr_Lines_Block.Content_Lines)
                Group_Dict[group].Child_Lines_Blocks.append(Curr_Lines_Block)
            else:
                group += 1

    if len(Group_Dict) == 0:
        return

    # consider defs/classes on continuous indent
    Group_List: tp.List[tp.List[int, SomeGroup]]
    Group_List = list(Group_Dict.items())

    subgroup = 0
    Subgroup_Dict: tp.Dict[tp.List[SomeGroup]] = {}

    Subgroup_Dict[subgroup] = [Group_List[0][-1]]
    for i in range(1, len(Group_List)):
        if Group_List[i][0] != Group_List[i - 1][0] + 1:
            subgroup += 1

        if subgroup not in Subgroup_Dict:
            Subgroup_Dict[subgroup] = []

        Subgroup_Dict[subgroup].append(Group_List[i][-1])

    # calibrate before after of parent
    Some_Group_List: tp.List[SomeGroup]
    for Some_Group_List in Subgroup_Dict.values():
        if len(Some_Group_List) >= 2:
            for i in range(len(Some_Group_List) - 1):
                if (Some_Group_List[i].unpadded_height <= mode.STYLE.code_height.tiny) and (
                    Some_Group_List[i + 1].unpadded_height <= mode.STYLE.code_height.tiny
                ):
                    Some_Group_List[i].Parent_Lines_Block.after = 0
                    Some_Group_List[
                        i + 1
                    ].Parent_Lines_Block.before = mode.STYLE.no_empty_lines.between_short_code

    # calibrate before after of all childs
    for Some_Group in Group_Dict.values():
        if Some_Group.unpadded_height <= mode.STYLE.code_height.short:
            Some_Group.Parent_Lines_Block.after = 0
            for Child_Lines_Block in Some_Group.Child_Lines_Blocks:
                Child_Lines_Block.after = 0
                Child_Lines_Block.before = 0

        elif Some_Group.unpadded_height >= mode.STYLE.code_height.tall:
            Some_Group.Parent_Lines_Block.after = 1
