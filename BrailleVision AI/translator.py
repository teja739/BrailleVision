# translator.py

from statistics import median

import numpy as np

from braille_map import BRAILLE_MAP


LEFT_DOTS = {0: 1, 1: 2, 2: 3}
RIGHT_DOTS = {0: 4, 1: 5, 2: 6}


def pattern_to_text(patterns):
    result = ""

    for pattern in patterns:
        if pattern == " ":
            result += " "
        else:
            result += BRAILLE_MAP.get(tuple(sorted(pattern)), "?")

    return result


def _cluster_values(values, gap_threshold, max_groups=None):
    if not values:
        return []

    groups = [[float(values[0])]]

    for value in values[1:]:
        value = float(value)
        if value - groups[-1][-1] <= gap_threshold:
            groups[-1].append(value)
        else:
            groups.append([value])

    if max_groups:
        while len(groups) > max_groups:
            merge_index = min(
                range(len(groups) - 1),
                key=lambda index: abs(np.mean(groups[index + 1]) - np.mean(groups[index])),
            )
            groups[merge_index].extend(groups.pop(merge_index + 1))
            groups[merge_index].sort()

    return groups


def _group_dots_by_axis(dots, axis, gap_threshold, max_groups=None):
    sorted_dots = sorted(dots, key=lambda item: item[axis])
    groups = []

    for dot in sorted_dots:
        if not groups or dot[axis] - groups[-1][-1][axis] > gap_threshold:
            groups.append([dot])
        else:
            groups[-1].append(dot)

    if max_groups:
        while len(groups) > max_groups:
            merge_index = min(
                range(len(groups) - 1),
                key=lambda index: abs(
                    _mean_axis(groups[index + 1], axis) - _mean_axis(groups[index], axis)
                ),
            )
            groups[merge_index].extend(groups.pop(merge_index + 1))
            groups[merge_index].sort(key=lambda item: item[axis])

    return groups


def _mean_axis(dots, axis):
    return float(np.mean([dot[axis] for dot in dots]))


def _estimate_row_groups(dots):
    y_values = sorted(dot[1] for dot in dots)
    y_span = max(y_values) - min(y_values) if len(y_values) > 1 else 0
    row_gap = max(9.0, min(24.0, y_span / 5.0 if y_span else 12.0))
    groups = _group_dots_by_axis(dots, axis=1, gap_threshold=row_gap, max_groups=3)
    centers = [_mean_axis(group, axis=1) for group in groups]
    return groups, centers


def _estimate_column_groups(dots):
    x_values = sorted(dot[0] for dot in dots)
    x_span = max(x_values) - min(x_values) if len(x_values) > 1 else 0
    column_gap = max(9.0, min(18.0, x_span / 35.0 if x_span else 12.0))
    groups = _group_dots_by_axis(dots, axis=0, gap_threshold=column_gap)
    centers = [_mean_axis(group, axis=0) for group in groups]
    return groups, centers


def _estimate_cell_split(column_centers, row_centers):
    if len(column_centers) <= 1:
        return 64.0

    gaps = [b - a for a, b in zip(column_centers, column_centers[1:])]
    sorted_gaps = sorted(gaps)

    if len(sorted_gaps) >= 2:
        for index in range(len(sorted_gaps) - 1):
            small_gap = sorted_gaps[index]
            large_gap = sorted_gaps[index + 1]

            if large_gap / max(small_gap, 1.0) >= 1.18:
                return (small_gap + large_gap) / 2.0

    if len(row_centers) >= 2:
        row_spacings = [b - a for a, b in zip(row_centers, row_centers[1:])]
        return median(row_spacings) * 1.15

    return 64.0


def _split_columns_into_cells(column_groups, column_centers, row_centers):
    if not column_groups:
        return []

    split_gap = _estimate_cell_split(column_centers, row_centers)
    cells = [[column_groups[0]]]

    for index in range(1, len(column_groups)):
        previous_center = _mean_axis(column_groups[index - 1], axis=0)
        current_center = _mean_axis(column_groups[index], axis=0)

        if current_center - previous_center <= split_gap and len(cells[-1]) < 2:
            cells[-1].append(column_groups[index])
        else:
            cells.append([column_groups[index]])

    return cells


def _nearest_index(value, centers):
    return min(range(len(centers)), key=lambda index: abs(value - centers[index]))


def dots_to_text(dots):
    """
    Convert detected dot centers into English text.

    This assumes a cropped, mostly horizontal six-dot Braille word. It is tuned
    for hackathon demos and generated samples, then can be improved for harder
    camera angles later.
    """

    if not dots:
        return {
            "text": "",
            "patterns": [],
            "cells": [],
            "confidence": 0.0,
        }

    clean_dots = [(int(x), int(y)) for x, y in dots]
    _, row_centers = _estimate_row_groups(clean_dots)
    column_groups, column_centers = _estimate_column_groups(clean_dots)
    cells_by_columns = _split_columns_into_cells(column_groups, column_centers, row_centers)

    cells = []
    patterns = []

    for cell_columns in cells_by_columns:
        pattern = set()
        column_centers_in_cell = [_mean_axis(column, axis=0) for column in cell_columns]
        cell_dots = [dot for column in cell_columns for dot in column]

        for column_index, column in enumerate(cell_columns):
            dot_lookup = LEFT_DOTS if column_index == 0 else RIGHT_DOTS

            for dot in column:
                row_index = _nearest_index(dot[1], row_centers)
                row_index = min(row_index, 2)
                pattern.add(dot_lookup[row_index])

        pattern_tuple = tuple(sorted(pattern))
        letter = BRAILLE_MAP.get(pattern_tuple, "?")

        cells.append(
            {
                "pattern": pattern_tuple,
                "letter": letter,
                "x": min(column_centers_in_cell),
                "dots": cell_dots,
            }
        )
        patterns.append(pattern_tuple)

    if len(cells) > 1:
        starts = [cell["x"] for cell in cells]
        start_gaps = [b - a for a, b in zip(starts, starts[1:])]
        normal_pitch = min(start_gaps) if start_gaps else 0
    else:
        normal_pitch = 0

    output_parts = []
    output_patterns = []

    for index, cell in enumerate(cells):
        if index > 0 and normal_pitch:
            gap = cell["x"] - cells[index - 1]["x"]
            if gap > normal_pitch * 1.55:
                output_parts.append(" ")
                output_patterns.append(" ")

        output_parts.append(cell["letter"])
        output_patterns.append(cell["pattern"])

    text = "".join(output_parts)
    recognized = sum(1 for char in text if char not in {"?", " "})
    total = sum(1 for char in text if char != " ")
    confidence = recognized / total if total else 0.0

    return {
        "text": text,
        "patterns": output_patterns,
        "cells": cells,
        "confidence": confidence,
    }
