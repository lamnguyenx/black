from .black import *
from .super_alignments.format_super_alignment import format_super_alignment
from .super_alignments.format_super_brackets import format_super_brackets
from .super_alignments.format_super_empty_lines import format_super_empty_lines
from .super_alignments.nested_show_level import nested_node_show_level
from .super_alignments.jot_line import jot_line


def _format_str_once__BLAX(
    src_contents: str, *, mode: Mode, lines: Collection[Tuple[int, int]] = ()
) -> str:
    # print('_format_str_once__BLAX')
    src_node = lib2to3_parse(src_contents.lstrip(), mode.target_versions)
    dst_blocks: List[LinesBlock] = []
    if mode.target_versions:
        versions = mode.target_versions
    else:
        future_imports = get_future_imports(src_node)
        versions = detect_target_versions(src_node, future_imports=future_imports)

    context_manager_features = {
        feature
        for feature in {Feature.PARENTHESIZED_CONTEXT_MANAGERS}
        if supports_feature(versions, feature)
    }
    normalize_fmt_off(src_node, mode, lines)
    if lines:
        # This should be called after normalize_fmt_off.
        convert_unchanged_lines(src_node, lines)

    line_generator = LineGenerator(mode=mode, features=context_manager_features)
    elt = EmptyLineTracker(mode=mode)
    split_line_features = {
        feature
        for feature in {Feature.TRAILING_COMMA_IN_CALL, Feature.TRAILING_COMMA_IN_DEF}
        if supports_feature(versions, feature)
    }
    block: Optional[LinesBlock] = None

    # original visit
    src_node = nested_node_show_level(src_node)

    for i, current_line in enumerate(line_generator.visit(src_node)):

        block = elt.maybe_empty_lines(current_line)
        dst_blocks.append(block)
        block.content_lines = []
        for j, line in enumerate(
            transform_line(current_line, mode=mode, features=split_line_features)
        ):
            if line.is_import and line.leaves[-1].value.endswith("("):
                line.leaves[-1].value += "\n"
            block.content_lines.append(line)

    format_super_alignment(dst_blocks, mode=mode)
    format_super_empty_lines(dst_blocks, "def", mode=mode)
    format_super_empty_lines(dst_blocks, "class", mode=mode)

    if dst_blocks:
        dst_blocks[-1].after = 0
    dst_contents = []
    for block in dst_blocks:
        dst_contents.extend(block.all_lines())
    if not dst_contents:
        # Use decode_bytes to retrieve the correct source newline (CRLF or LF),
        # and check if normalized_content has more than one line
        normalized_content, _, newline = decode_bytes(src_contents.encode("utf-8"))
        if "\n" in normalized_content:
            return newline
        return ""

    out = "".join(dst_contents)
    return out
