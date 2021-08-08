import random
from math import ceil
from math import sqrt
from statistics import mean
from typing import List
from typing import Tuple

import numpy as np

from configs.make_config import Config
from optimizer.lib.defs import Box
from optimizer.lib.defs import LineSegment
from optimizer.lib.defs import Lyrics
from optimizer.lib.defs import Point


def len_of_text_list(text: Tuple[str, ...]) -> int:
    text_pad = 1
    length = text_pad  # text pad for left side
    for word in text:
        length += len(word)
        length += 1  # added for a space between word
    length += text_pad  # adding text pad for right side
    return length


def find_font_size_and_pattern(lyrics_box: Box, lyrics: Lyrics):
    pattern = int(lyrics_box.width / lyrics_box.height) + 1
    if pattern < 2:
        pattern = 2
    elif pattern > 5:
        pattern = 5
    max_width = 0
    num_lines = ceil(len(lyrics.text) / pattern)
    for i in range(0, len(lyrics.text), pattern):
        length = len(" ".join(lyrics.text[i : i + pattern]))
        if length > max_width:
            max_width = length
    max_width += 2
    font_size_init = int(lyrics_box.height / (num_lines + 1))
    for size in range(font_size_init, int(font_size_init / 4), -1):
        if (size / 2) * max_width < lyrics_box.width:
            return size, pattern

    return False, False


def get_expected_box_dims(lyrics: Lyrics, font_size: int, form: int) -> Tuple[int, int]:

    n_words = len(lyrics.text)
    lengths_of_lines = []
    if n_words > form:
        for i in range(0, n_words - form, form):
            if i + form < n_words:
                last_index = i + form
            else:
                last_index = n_words - 1
            lengths_of_lines.append(len_of_text_list(lyrics.text[i:last_index]))
    else:
        lengths_of_lines.append(len_of_text_list(lyrics.text))

    # max length will never be zero
    expected_width = int(max(lengths_of_lines) * font_size / 2)
    expected_height = round(n_words / form) * font_size

    return expected_width, expected_height


def text_fits_box(expected_width: int, expected_height: int, box: Box) -> bool:

    return (
        box.width > expected_width
        and box.height > expected_height
        and box.width / box.height > 1
    )


def is_lyrics_box_big_enough_to_be_readable(
    canvas_shape: Tuple[int, int], lyrics_box: Box
) -> bool:

    # width and height of lyrics-box should be greater than 20% of width & 10% height of the image
    return (
        lyrics_box.width > 0.20 * canvas_shape[1]  # width
        and lyrics_box.height > 0.10 * canvas_shape[0]  # height
    )


def get_overlap_with_mask(image: np.ndarray, lyrics_box: Box, padding: int):
    box_array = np.ones(shape=[lyrics_box.height + padding, lyrics_box.width + padding])

    cropped_image_array = image[
        lyrics_box.vertex_1.y - padding // 2 : lyrics_box.vertex_3.y + padding // 2,
        lyrics_box.vertex_1.x - padding // 2 : lyrics_box.vertex_3.x + padding // 2,
    ]

    score = (box_array * cropped_image_array).sum()

    return score


def get_norm_distance_from_image_edges(
    canvas_shape: Tuple[int, int], box: Box
) -> List[float]:

    canvas_diag_length = sqrt(canvas_shape[0] ** 2 + canvas_shape[1] ** 2)

    distance_edge_left = box.vertex_1.x / canvas_diag_length
    distance_edge_right = (canvas_shape[1] - box.vertex_3.x) / canvas_diag_length
    distance_edge_top = box.vertex_1.y / canvas_diag_length
    distance_edge_bottom = (canvas_shape[0] - box.vertex_3.y) / canvas_diag_length

    return [
        distance_edge_left,
        distance_edge_right,
        distance_edge_top,
        distance_edge_bottom,
    ]


def get_combined_box(boxes: Tuple[Box, ...]) -> Box:

    min_x = min([box.vertex_1.x for box in boxes])
    min_y = min([box.vertex_1.y for box in boxes])

    max_x = max([box.vertex_3.x for box in boxes])
    max_y = max([box.vertex_3.y for box in boxes])

    new_box = Box(
        first_diagonal_coords=Point((min_x, min_y)),
        second_diagonal_coords=Point((max_x, max_y)),
    )

    return new_box


def get_nearness_to_preferred_centre(centre_1: Point, centre_2: Point) -> float:
    line_seg = LineSegment(centre_1, centre_2)

    return line_seg.length


def get_preferred_centre(boxes: Tuple[Box, ...], image: np.ndarray) -> Point:

    # check the spread of boxes
    combi_box = get_combined_box(boxes)

    total_area_of_all_boxes = sum([box.area for box in boxes])

    available_area_in_between_boxes = combi_box.area - total_area_of_all_boxes

    # if all boxes together occupy > 50% of the image area and more than 50% of the area in between boxes is available
    if (
        combi_box.area / (image.shape[0] * image.shape[1]) > 0.5
        and available_area_in_between_boxes / combi_box.area > 0.5
    ):
        # search for an optimal location within the boxes
        list_of_centres = [box.centre for box in boxes]

        naive_centre_x = mean([centre.x for centre in list_of_centres])
        naive_centre_y = mean([centre.y for centre in list_of_centres])

        naive_centre = Point(coords=(naive_centre_x, naive_centre_y))

        # if the naive centre is inside any box
        if any([box.is_enclosing(naive_centre) for box in boxes]):
            # TODO: develop a better logic here !
            # then return the centre of the imaage as the preferred centre
            preferred_centre = Point(coords=(image.shape[0] // 2, image.shape[1] // 2))
        else:
            preferred_centre = naive_centre
    else:
        # search for an optimal location in the area of the image where there are no boxes

        print("do something else")

    return preferred_centre


def get_overlapping_area(box_1: Box, box_2: Box) -> int:

    if not box_1.is_overlapping(box_2):
        return 0

    # make overlap box
    x1 = max(box_1.vertex_1.x, box_2.vertex_1.x)
    y1 = max(box_1.vertex_1.y, box_2.vertex_1.y)
    x3 = min(box_1.vertex_3.x, box_2.vertex_3.x)
    y3 = min(box_1.vertex_3.y, box_2.vertex_3.y)

    try:
        return Box(
            first_diagonal_coords=Point((x1, y1)),
            second_diagonal_coords=Point((x3, y3)),
        ).area
    except AssertionError:
        # there are only 2 cases when the above fail -
        # if the two boxes touch each other with a line or a point
        return 0


def get_bottom_box(conf: Config) -> Tuple[int, int, int, int]:
    x = conf.img_width // 2
    y = int(conf.img_height * 0.90)
    x1 = x - conf.img_width // 4
    y1 = y - int(0.15 * conf.img_height)
    x3 = x + conf.img_width // 4
    y3 = y

    return x1, y1, x3, y3


def is_box_big_enough_to_be_made_smaller_for_variation(
    x1, y1, x3, y3, canvas_shape: Tuple[int, int]
) -> bool:

    return (x3 - x1) * (y3 - y1) / (canvas_shape[0] * canvas_shape[1]) > 0.35


def add_variation(
    x1, y1, x3, y3, canvas_shape: Tuple[int, int], small_box_probability: float = 0.5
) -> Tuple[int, int, int, int]:

    if (
        is_box_big_enough_to_be_made_smaller_for_variation(x1, y1, x3, y3, canvas_shape)
        and random.choices(
            [True, False], weights=[small_box_probability, 1 - small_box_probability]
        )[0]
    ):
        x1_ = x1 + 0.1 * (x3 - x1)
        x3_ = x3 - 0.1 * (x3 - x1)
        y1_ = y1 + 0.1 * (y3 - y1)
        y3_ = y3 - 0.1 * (y3 - y1)
        return int(round(x1_)), int(round(y1_)), int(round(x3_)), int(round(y3_))
    else:
        return x1, y1, x3, y3
