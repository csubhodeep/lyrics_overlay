from functools import partial
from math import sqrt
from multiprocessing import Pool
from pathlib import Path
from statistics import variance
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from configs.make_config import Config
from optimizer.lib.defs import Box
from optimizer.lib.defs import Lyrics
from optimizer.lib.defs import Point
from optimizer.utils.params import LossFunctionParameters
from optimizer.utils.params import OptimizerParameters
from optimizer.utils.utils import get_distance_from_image_edges
from optimizer.utils.utils import is_box_big_enough

# import matplotlib.pyplot as plt # noqa


def get_loss(
    x, canvas_shape: Tuple[int, int], forbidden_zones: Tuple[Box, ...], text: Lyrics
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
    except AssertionError:
        return LossFunctionParameters.WRONG_COORDINATE_COST

    if any([lyrics_box.is_overlapping(zone) for zone in forbidden_zones]):
        return LossFunctionParameters.OVERLAPPING_COST

    if not is_box_big_enough(canvas_shape=canvas_shape, lyrics_box=lyrics_box):
        return LossFunctionParameters.SMALL_BOX_COST

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

    # we get distance from 3 edges of the image. and we believe that obviously the best box might be close to
    # one of the edge. so lets not optimize for all 4 edges. only optimize for 3 edges. (infact we are ingnoring 1 out
    # 2 side edges )
    if len(forbidden_zones) == 1:
        dist_of_f_zone_from_left_edge = forbidden_zones[0].vertex_1.x - 0
        dist_of_f_zone_from_right_edge = canvas_shape[1] - forbidden_zones[0].vertex_3.x
        if dist_of_f_zone_from_left_edge < dist_of_f_zone_from_right_edge:
            distance_edges.pop(0)
        else:
            distance_edges.pop(1)
    else:
        raise Exception("Optimizer can only with one forbidden zone")

    all_distances = tuple(distance_edges) + distance_persons

    if min(all_distances) < LossFunctionParameters.MIN_DISTANCE_THRESHOLD:
        return LossFunctionParameters.MIN_DISTANCE_COST
    else:
        return sqrt(variance(all_distances)) + 1 / lyrics_box.area


def get_optimal_boxes(row, conf: Config) -> Dict[str, Union[int, float]]:

    # if forbidden zone is an invalid zone...return centre of image
    if not (row["x1"] == row["y1"] == row["x3"] == row["y3"] == -1):
        persons = (
            Box(
                first_diagonal_coords=Point(coords=(row["x1"], row["y1"])),
                second_diagonal_coords=Point(coords=(row["x3"], row["y3"])),
            ),
        )

        lyrics = Lyrics(
            text=row["text"], start_time=row["start_time"], end_time=row["end_time"]
        )

        limits = (
            (0, conf.img_width),
            (0, conf.img_height),
            (0, conf.img_width),
            (0, conf.img_height),
        )

        res = differential_evolution(
            get_loss,
            bounds=limits,
            args=((conf.img_height, conf.img_width), persons, lyrics),
            popsize=OptimizerParameters.POPULATION_SIZE,
        )

        if res.success and res.fun < LossFunctionParameters.MAXIMUM_LOSS_THRESHOLD:
            x1 = int(round(res.x[0]))
            y1 = int(round(res.x[1]))
            x3 = int(round(res.x[2]))
            y3 = int(round(res.x[3]))
        else:
            x = conf.img_width // 2
            y = int(conf.img_height * 0.90)
            x1 = x - conf.img_width // 4
            y1 = y - int(0.10 * conf.img_height)
            x3 = x + conf.img_width // 4
            y3 = y
    else:
        x1 = int(0.20 * conf.img_width)
        y1 = int(0.20 * conf.img_height)
        x3 = int(0.80 * conf.img_width)
        y3 = int(0.80 * conf.img_height)

    return {
        "x1_opti": x1,
        "y1_opti": y1,
        "x3_opti": x3,
        "y3_opti": y3,
        "start_time": row["start_time"],
        "end_time": row["end_time"],
    }


def get_image_height_and_width(conf: Config) -> Config:
    # TODO: make it better

    path_to_files = Path.cwd().joinpath(f"data/pre_processor_output/{conf.run_id}")

    for item in path_to_files.iterdir():
        if item.name.endswith("npy"):
            with open(str(item), "rb") as f:
                frame = np.load(f)
                break
        else:
            continue

    print(frame.shape)

    conf.img_height, conf.img_width = frame.shape[0], frame.shape[1]

    return conf


def optimize(conf: Config) -> bool:

    conf = get_image_height_and_width(conf)

    input_file_path = (
        Path.cwd().joinpath(conf.input_data_path).joinpath(f"{conf.run_id}.feather")
    )
    output_file_path = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.feather")
    )

    df_input = pd.read_feather(input_file_path)
    inps = []
    for idx, row in df_input.iterrows():
        inps.append(row)

    # using multiprocessing.Pool - takes 18s
    with Pool() as p:
        res = p.map(partial(get_optimal_boxes, conf=conf), inps)

    df_opti = pd.DataFrame(res)
    df_input = df_input.merge(df_opti, on=["start_time", "end_time"])

    # NORMAL pandas apply() - takes 33s kept here only for debugging/testing
    # df_input[["x1_opti", "y1_opti", "x3_opti", "y3_opti"]] = df_input.apply(
    #     get_optimal_boxes, axis=1, args=(conf,), result_type="expand"
    # )

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
        run_id="1ffadc62-6f97-43e1-a51a-029110d0cb84",
    )

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

    # binary_mask[y1:y3, x1:x3]

    # # case 0
    # binary_mask = np.zeros([100, 100])
    #
    # persons = ()

    # # case 1
    # binary_mask = np.zeros([100, 100])
    # binary_mask[40:60, 60:80] = 1
    #
    # persons = (
    # 		Box(first_diagonal_coords=Point(coords=(60,40)), second_diagonal_coords=Point(coords=(80,60))),
    # 	)

    # # case 2
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

    # # case 3
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

    # # case 4
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
