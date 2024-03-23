#!/bin/bash

# Copyright 2024 (author: lamnguyenx)

# DIRECTION='
# Add the following lines to your .bashrc:

#     # %--start--
#     # per-project environment
#     if [[ -f .project.sh ]]; then
#         source .project.sh
#     fi
#     # %--end--

# '
SWD="$(dirname "$(realpath "$BASH_SOURCE")")"


### GENERAL
export LOG_LEVEL="${LOG_LEVEL:-"INFO"}"
export PROJECT_ROOT="$SWD"


PYTHONPATH_PLUS="${SWD}/src:${SWD}/exp/__pylinks__"

export PYTHONPATH="$PYTHONPATH_PLUS:${PYTHONPATH:-""}"
export MYPYPATH=$PYTHONPATH
