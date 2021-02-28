from math import sqrt
from pathlib import Path
from statistics import variance
from typing import Iterable
from typing import Tuple

import pandas as pd
from scipy.optimize import differential_evolution

from configs.make_config import Config
from optimizer.lib.defs import Box
from optimizer.lib.defs import Lyrics
from optimizer.lib.defs import Point
from optimizer.utils.params import Costs
from optimizer.utils.params import FontLimits
from optimizer.utils.utils import get_distance_from_image_edges
from optimizer.utils.utils import get_expected_box_dims
from optimizer.utils.utils import text_fits_box

# import matplotlib.pyplot as plt # noqa


def get_loss(
    x, canvas_shape: Tuple[int, int], forbidden_zones: Iterable[Box], text: Lyrics
) -> float:
    """
    Args:
        x:
        canvas_shape:
        forbidden_zones:
        text:

    Returns:

    """
    try:
        lyrics_box = Box(
            first_diagonal_coords=Point(coords=(x[0], x[1])),
            second_diagonal_coords=Point(coords=(x[2], x[3])),
        )
    except AssertionError as ex:
        return Costs.WRONG_COORDINATE_COST

    if any([lyrics_box.is_overlapping(zone) for zone in forbidden_zones]):
        return Costs.OVERLAPPING_COST

    expected_width, expected_height = get_expected_box_dims(
        lyrics=text, font_size=int(round(x[4])), form=FontLimits.FORM_LIMIT[1]
    )

    is_fit = text_fits_box(expected_width, expected_height, lyrics_box)

    if not is_fit:
        return Costs.TEXT_NOT_FITTING_COST

    # # include the following:
    # # distance from all person-boxes - w1

    # # iterate over all the edges of all person-boxes and find the distances of them from the lyrics-box
    if len(forbidden_zones):
        distance_persons = tuple(
            [lyrics_box.get_distance_from(zone) for zone in forbidden_zones]
        )
    else:
        distance_persons = tuple([])

    # # distance from all 4 edges - w2
    distance_edges = get_distance_from_image_edges(canvas_shape, lyrics_box)

    if len(forbidden_zones) == 1:
        dist_of_f_zone_from_left_edge = forbidden_zones[0].vertex_1.x - 0
        dist_of_f_zone_from_right_edge = canvas_shape[1] - forbidden_zones[0].vertex_3.x
        if dist_of_f_zone_from_left_edge < dist_of_f_zone_from_right_edge:
            distance_edges.pop(0)
        else:
            distance_edges.pop(1)

    all_distances = tuple(distance_edges) + distance_persons

    if min(all_distances) < 20:
        return Costs.MIN_DISTANCE_COST
    else:
        return sqrt(variance(all_distances)) + 1 / lyrics_box.area


def get_optimal_boxes(row, conf: Config):

    persons = (
        Box(
            first_diagonal_coords=Point(coords=(row["x1"], row["y1"])),
            second_diagonal_coords=Point(coords=(row["x3"], row["y3"])),
        ),
    )

    lyrics = Lyrics(row["text"])

    limits = (
        (0, conf.img_width),
        (0, conf.img_height),
        (0, conf.img_width),
        (0, conf.img_height),
        FontLimits.FONT_SIZE_LIMIT,
        # (1, 2),
    )

    res = differential_evolution(
        get_loss,
        bounds=limits,
        args=((conf.img_height, conf.img_width), persons, lyrics),
        popsize=100,
    )

    if res.success:
        return (
            int(round(res.x[0])),
            int(round(res.x[1])),
            int(round(res.x[2])),
            int(round(res.x[3])),
            int(round(res.x[4])),
            FontLimits.FORM_LIMIT[1],
        )
    else:
        expected_width, expected_height = get_expected_box_dims(
            lyrics, font_size=5, form=2
        )
        x = conf.img_width // 2
        y = int(conf.img_height * 0.9)
        x1 = x - expected_width // 2
        y1 = y - expected_height // 2
        x3 = x1 + expected_width
        y3 = y1 + expected_height
        return x1, y1, x3, y3, FontLimits.FONT_SIZE_LIMIT[1], FontLimits.FORM_LIMIT[1]


def optimize(conf: Config) -> bool:

    input_file_path = (
        Path.cwd().joinpath(conf.input_data_path).joinpath(f"{conf.run_id}.feather")
    )
    output_file_path = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.feather")
    )

    df_input = pd.read_feather(input_file_path)

    df_input[
        ["x1_opti", "y1_opti", "x3_opti", "y3_opti", "font_size", "form"]
    ] = df_input.apply(get_optimal_boxes, axis=1, args=(conf,), result_type="expand")

    df_input[["x1_opti", "y1_opti", "x3_opti", "y3_opti"]] = df_input[
        ["x1_opti", "y1_opti", "x3_opti", "y3_opti"]
    ].astype(int)

    df_input.to_feather(output_file_path)

    return True


if __name__ == "__main__":

    config = Config(
        output_data_path="../data/optimizer_output",
        input_data_path="../data/splitter_output",
        img_width=739,
        img_height=416,
    )
    config.set_run_id(run_id="a1945f8a-6fbf-4686-a1fe-486fcfed1590")

    optimize(conf=config)

    # lyrics = Lyrics("I love you I love you I love you I love you")
    #
    # limits = (
    # 	(0, 100),
    # 	(0, 100),
    # 	(0, 100),
    # 	(0, 100),
    # 	(1, 5)
    # )

    # #binary_mask[y1:y3, x1:x3]

    ### case 0
    # binary_mask = np.zeros([100, 100])
    #
    # persons = ()

    ### case 1
    # binary_mask = np.zeros([100, 100])
    # binary_mask[40:60, 60:80] = 1
    #
    # persons = (
    # 		Box(first_diagonal_coords=Point(coords=(60,40)), second_diagonal_coords=Point(coords=(80,60))),
    # 	)

    ### case 2
    # binary_mask = np.zeros([100, 100])
    #
    # binary_mask[10:30, 10:30] = 1
    # binary_mask[40:60, 40:60] = 1
    # binary_mask[70:90, 70:90] = 1
    #
    # persons = (
    # 	Box(first_diagonal_coords=Point(coords=(10, 10)), second_diagonal_coords=Point(coords=(30, 30))),
    # 	Box(first_diagonal_coords=Point(coords=(40, 40)), second_diagonal_coords=Point(coords=(60, 60))),
    # 	Box(first_diagonal_coords=Point(coords=(70, 70)), second_diagonal_coords=Point(coords=(90, 90)))
    # )

    ### case 3
    # binary_mask = np.zeros([100, 100])
    # binary_mask[20:60, 10:40] = 1
    # binary_mask[10:30, 65:85] = 1
    # binary_mask[70:90, 50:70] = 1
    #
    # persons = (
    # 	Box(first_diagonal_coords=Point(coords=(10, 20)), second_diagonal_coords=Point(coords=(40, 60))),
    # 	Box(first_diagonal_coords=Point(coords=(65, 10)), second_diagonal_coords=Point(coords=(85, 30))),
    # 	Box(first_diagonal_coords=Point(coords=(50, 70)), second_diagonal_coords=Point(coords=(70, 90)))
    # )

    ### case 4
    # binary_mask = np.zeros([100, 100])
    # binary_mask[20:80, 10:25] = 1
    # binary_mask[20:80, 47:52] = 1
    # binary_mask[20:80, 70:85] = 1
    #
    # persons = (
    # 	Box(first_diagonal_coords=Point(coords=(10, 20)), second_diagonal_coords=Point(coords=(25, 80))),
    # 	Box(first_diagonal_coords=Point(coords=(47, 20)), second_diagonal_coords=Point(coords=(52, 80))),
    # 	Box(first_diagonal_coords=Point(coords=(70, 20)), second_diagonal_coords=Point(coords=(85, 80)))
    # )

    # TODO: why not exclude certain solutions before triggering the opti algo for faster convergence

    # res = differential_evolution(get_loss,
    # 							 bounds=limits,
    # 							 args=(binary_mask.shape, persons, lyrics),
    # 							 popsize=100
    # 							 )
    #
    # if res.success:
    # 	optimal_box = Box(first_diagonal_coords=Point((res.x[0], res.x[1])),
    # 				  second_diagonal_coords=Point((res.x[2], res.x[3])))
    #
    # 	plt.imshow(optimal_box.overlay_on_image(binary_mask))
    # 	plt.show()
    # 	print(res)
