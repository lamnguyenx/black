#!/usr/bin/env python3

# Copyright 2023 (author: lamnguyenx)


# parent
from ..blib2to3.pytree import Node, Leaf



### FUNCTIONS
def nested_node_show_level(
    Inp_Node: Node,
    code_level: int = 0,
):
    if type(Inp_Node) == Node:
        Inp_Node.code_level = code_level
        for i in range(len(Inp_Node.children)):
            Inp_Node.children[i] = nested_node_show_level(
                Inp_Node=Inp_Node.children[i],
                code_level=code_level + 1,
            )

    elif type(Inp_Node) == Leaf:
        Inp_Node.code_level = code_level

    return Inp_Node
