#!/usr/bin/env python3

# Copyright 2024 (author: lamnguyenx)


# technical
import typing as tp
from dataclasses import dataclass


# local
from .black.lines import *
from .super_alignments.const import STYLE_LAMNGUYENX



### CLASSES
@dataclass
class Line__BLAX(Line):

    def get_all_properties(self) -> dict:
        return {
            'is_comment'                 : self.is_comment,
            'is_decorator'               : self.is_decorator,
            'is_import'                  : self.is_import,
            'is_with_or_async_with_stmt' : self.is_with_or_async_with_stmt,
            'is_class'                   : self.is_class,
            'is_stub_class'              : self.is_stub_class,
            'is_def'                     : self.is_def,
            'is_stub_def'                : self.is_stub_def,
            'is_class_paren_empty'       : self.is_class_paren_empty,
            'is_triple_quoted_string'    : self.is_triple_quoted_string,
            'is_chained_assignment'      : self.is_chained_assignment,
        }

    @property
    def is_not_special(self) -> bool:
        return not any(self.get_all_properties().values())


    def __repr__(self) -> str:
        return ' '.join(map(str, self.leaves))



@dataclass
class LinesBlock__BLAX(LinesBlock):

    def all_lines(self) -> List[str]:
        empty_line = str(Line(mode=self.mode))
        prefix = make_simple_prefix(self.before, self.form_feed, empty_line)
        return [prefix] + [str(L) for L in self.content_lines]+ [empty_line * self.after]



@dataclass
class EmptyLineTracker__BLAX(EmptyLineTracker):

    def _maybe_empty_lines(self, current_line: Line) -> Tuple[int, int]:
        max_allowed = 1
        if current_line.depth == 0:
            max_allowed = 1 if self.mode.is_pyi else STYLE_LAMNGUYENX.no_empty_lines.default
        if current_line.leaves:
            # Consume the first leaf's extra newlines.
            first_leaf = current_line.leaves[0]
            before = first_leaf.prefix.count("\n")
            before = min(before, max_allowed)
            first_leaf.prefix = ""
        else:
            before = 0

        user_had_newline = bool(before)
        depth = current_line.depth

        previous_def = None
        while self.previous_defs and self.previous_defs[-1].depth >= depth:
            previous_def = self.previous_defs.pop()


        if previous_def is not None:
            assert self.previous_line is not None
            if self.mode.is_pyi:
                if depth and not current_line.is_def and self.previous_line.is_def:
                    # Empty lines between attributes and methods should be preserved.
                    before = 1 if user_had_newline else 0
                elif (
                    Preview.blank_line_after_nested_stub_class in self.mode
                    and previous_def.is_class
                    and not previous_def.is_stub_class
                ):
                    before = 1
                elif depth:
                    before = 0
                else:
                    before = 1
            else:
                if depth:
                    before = 1
                elif (
                    not depth
                    and previous_def.depth
                    and current_line.leaves[-1].type == token.COLON
                    and (
                        current_line.leaves[0].value
                        not in ("with", "try", "for", "while", "if", "match")
                    )
                ):
                    # We shouldn't add two newlines between an indented function and
                    # a dependent non-indented clause. This is to avoid issues with
                    # conditional function definitions that are technically top-level
                    # and therefore get two trailing newlines, but look weird and
                    # inconsistent when they're followed by elif, else, etc. This is
                    # worse because these functions only get *one* preceding newline
                    # already.
                    before = 1
                else:
                    before = STYLE_LAMNGUYENX.no_empty_lines.default

        if current_line.is_decorator or current_line.is_def or current_line.is_class:
            return self._maybe_empty_lines_for_class_or_def(
                current_line, before, user_had_newline
            )

        if (
            self.previous_line
            and self.previous_line.is_import
            and not current_line.is_import
            and not current_line.is_fmt_pass_converted(first_leaf_matches=is_import)
            and depth == self.previous_line.depth
        ):
            if STYLE_LAMNGUYENX.__name__ == 'STYLE_LAMNGUYENX':
                for prev_leave in self.previous_line.leaves:
                    for curr_leave in current_line.leaves:
                        if curr_leave.value in prev_leave.value:
                            return (before or 0), 0

            return (before or 1), 0

        if (
            self.previous_line
            and self.previous_line.is_class
            and current_line.is_triple_quoted_string
        ):
            if Preview.no_blank_line_before_class_docstring in current_line.mode:
                return 0, 1
            return before, 1

        # In preview mode, always allow blank lines, except right before a function
        # docstring
        is_empty_first_line_ok = (
            Preview.allow_empty_first_line_in_block in current_line.mode
            and (
                not is_docstring(current_line.leaves[0])
                or (
                    self.previous_line
                    and self.previous_line.leaves[0]
                    and self.previous_line.leaves[0].parent
                    and not is_funcdef(self.previous_line.leaves[0].parent)
                )
            )
        )

        if (
            self.previous_line
            and self.previous_line.opens_block
            and not is_empty_first_line_ok
        ):
            if (self.previous_line.is_class):
                return STYLE_LAMNGUYENX.no_empty_lines.opens_block, 0

            return 0, 0


        return before, 0



    def _maybe_empty_lines_for_class_or_def(  # noqa: C901
        self, current_line: Line, before: int, user_had_newline: bool
    ) -> Tuple[int, int]:
        if not current_line.is_decorator:
            self.previous_defs.append(current_line)
        if self.previous_line is None:
            # Don't insert empty lines before the first line in the file.
            return STYLE_LAMNGUYENX.no_empty_lines.between_class_and_first_one, 0

        if self.previous_line.is_decorator:
            if self.mode.is_pyi and current_line.is_stub_class:
                # Insert an empty line after a decorated stub class
                return 0, 1

            return 0, 0

        if self.previous_line.depth < current_line.depth and (
            self.previous_line.is_class or self.previous_line.is_def
        ):
            return STYLE_LAMNGUYENX.no_empty_lines.between_class_and_first_one, 0

        comment_to_add_newlines: Optional[LinesBlock] = None
        if (
            self.previous_line.is_comment
            and self.previous_line.depth == current_line.depth
            and before == 0
        ):
            slc = self.semantic_leading_comment
            if (
                slc is not None
                and slc.previous_block is not None
                and not slc.previous_block.original_line.is_class
                and not slc.previous_block.original_line.opens_block
                and slc.before <= 1
            ):
                comment_to_add_newlines = slc
            else:
                return 0, 0

        if self.mode.is_pyi:
            if current_line.is_class or self.previous_line.is_class:
                if self.previous_line.depth < current_line.depth:
                    newlines = 0
                elif self.previous_line.depth > current_line.depth:
                    newlines = 1
                elif current_line.is_stub_class and self.previous_line.is_stub_class:
                    # No blank line between classes with an empty body
                    newlines = 0
                else:
                    newlines = 1
            # Remove case `self.previous_line.depth > current_line.depth` below when
            # this becomes stable.
            #
            # Don't inspect the previous line if it's part of the body of the previous
            # statement in the same level, we always want a blank line if there's
            # something with a body preceding.
            elif (
                Preview.blank_line_between_nested_and_def_stub_file in current_line.mode
                and self.previous_line.depth > current_line.depth
            ):
                newlines = 1
            elif (
                current_line.is_def or current_line.is_decorator
            ) and not self.previous_line.is_def:
                if current_line.depth:
                    # In classes empty lines between attributes and methods should
                    # be preserved.
                    newlines = min(1, before)
                else:
                    # Blank line between a block of functions (maybe with preceding
                    # decorators) and a block of non-functions
                    newlines = 1
            elif self.previous_line.depth > current_line.depth:
                newlines = 1
            else:
                newlines = 0
        else:
            newlines = STYLE_LAMNGUYENX.no_empty_lines.between_class_methods if current_line.depth else STYLE_LAMNGUYENX.no_empty_lines.default
            # If a user has left no space after a dummy implementation, don't insert
            # new lines. This is useful for instance for @overload or Protocols.
            if (
                Preview.dummy_implementations in self.mode
                and self.previous_line.is_stub_def
                and not user_had_newline
            ):
                newlines = 0
        if comment_to_add_newlines is not None:
            previous_block = comment_to_add_newlines.previous_block
            if previous_block is not None:
                comment_to_add_newlines.before = (
                    max(comment_to_add_newlines.before, newlines) - previous_block.after
                )
                newlines = 0

        return newlines, 0
