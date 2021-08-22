import os
import random
from math import sqrt
from pathlib import Path
from typing import List
from typing import Tuple

import cv2
from wand.font import Font
from wand.image import Image

from configs.make_config import Config
from optimizer.lib.defs import Box
from optimizer.lib.defs import Point

if os.getenv("ENVIRONMENT") == "test":
    DEBUG = True
else:
    DEBUG = False


def len_of_text_list(text: Tuple[str, ...]) -> int:
    text_pad = 1
    length = text_pad  # text pad for left side
    for word in text:
        length += len(word)
        length += 1  # added for a space between word
    length += text_pad  # adding text pad for right side
    return length


def get_size_of_original_video(conf: Config) -> Tuple[int, int]:
    # TODO: make it better

    input_video_file_name = (
        Path.cwd().joinpath(conf.video_input_path).joinpath(f"{conf.run_id}.mp4")
    )

    cap = cv2.VideoCapture(str(input_video_file_name))

    while cap.isOpened():
        ret, frame = cap.read()
        break

    cap.release()
    cv2.destroyAllWindows()

    return frame.shape


def is_lyrics_box_big_enough_to_be_readable(
    canvas_shape: Tuple[int, int], lyrics_box: Box
) -> bool:

    # width and height of lyrics-box should be greater than 20% of width & 10% height of the image
    return (
        lyrics_box.width > 0.20 * canvas_shape[1]  # width
        and lyrics_box.height > 0.10 * canvas_shape[0]  # height
    )


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
    y1 = y - int(0.30 * conf.img_height)
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


def restore_scale_to_original_resolution(
    row, conf: Config, for_opti: bool = False
) -> Tuple[int, int, int, int]:

    if for_opti:
        x1 = "x1_opti"
        y1 = "y1_opti"
        x3 = "x3_opti"
        y3 = "y3_opti"
    else:
        x1 = "x1"
        y1 = "y1"
        x3 = "x3"
        y3 = "y3"

    if row[x1] == row[y1] == row[x3] == row[y3] == -1:
        return 0, 0, 0, 0

    box = Box(
        first_diagonal_coords=Point(coords=(row[x1], row[y1])),
        second_diagonal_coords=Point(coords=(row[x3], row[y3])),
    )

    box.resize(
        new_canvas_shape=conf.org_canvas_shape,
        old_canvas_shape=(conf.img_height, conf.img_width),
    )

    return (box.vertex_1.x, box.vertex_1.y, box.vertex_3.x, box.vertex_3.y)


def draw_text_inside_box(row, conf: Config, font_path: Path) -> None:

    lyrics_box = Box(
        first_diagonal_coords=Point(coords=(row["x1_opti"], row["y1_opti"])),
        second_diagonal_coords=Point(coords=(row["x3_opti"], row["y3_opti"])),
    )

    if DEBUG:
        font_name = random.choice(["Playlist_Script.otf", "Black.otf", "yatra_one.ttf"])
    else:
        font_name = conf.font_name

    transparent_canvas = Image(
        width=lyrics_box.width, height=lyrics_box.height, pseudo="xc:transparent"
    )
    left, top, width, height = 0, 0, lyrics_box.width, lyrics_box.height
    font = Font(str(font_path.joinpath(font_name)), color="black")
    transparent_canvas.caption(
        row["text"],
        left=left,
        top=top,
        width=width,
        height=height,
        font=font,
        gravity="center",
    )
    x_shadow_offset = min(int(0.008 * lyrics_box.width), 2)
    y_shadow_offset = min(int(0.008 * lyrics_box.height), 2)
    left, top, width, height = (
        x_shadow_offset,
        y_shadow_offset,
        lyrics_box.width,
        lyrics_box.height,
    )
    font = Font(str(font_path.joinpath(font_name)), color="white")
    transparent_canvas.caption(
        row["text"],
        left=left,
        top=top,
        width=width,
        height=height,
        font=font,
        gravity="center",
    )

    output_folder_path = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(conf.run_id)
    )
    output_folder_path.mkdir(exist_ok=True)

    transparent_canvas.save(
        filename=str(output_folder_path.joinpath(f"{row['start_time']}.png"))
    )
