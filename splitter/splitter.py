from pathlib import Path
from typing import List
from typing import Tuple

import pandas as pd

from configs.make_config import Config


def get_cut_points(df: pd.DataFrame) -> Tuple[float, ...]:

    # # TODO: implement IOU logic
    """IOU logic splits it further"""

    cut_frames = (df["frame"].max() + 1,)

    return cut_frames


def split_dfs(
    df: pd.DataFrame, cutting_frames: Tuple[float, ...]
) -> List[pd.DataFrame]:
    """This function gives us a list of slices of the input dataframe"""
    list_of_df = []
    cutting_frames_sorted = sorted(cutting_frames)
    for idx, frame in enumerate(cutting_frames_sorted):
        curr_spltting_point = frame
        if idx == 0:
            previous_splitting_point = 0.0
        else:
            previous_splitting_point = cutting_frames_sorted[idx - 1]
        list_of_df.append(
            df.loc[
                (df["frame"] < curr_spltting_point)
                & (df["frame"] >= previous_splitting_point)
            ]
        )

    return list_of_df


def do_union(df: pd.DataFrame) -> Tuple[int, int, int, int]:

    x1 = df["x1"].min()
    x3 = df["x3"].max()
    y1 = df["y1"].min()
    y3 = df["y3"].max()

    return x1, y1, x3, y3


def split(conf: Config) -> bool:
    file_name = conf.run_id

    input_f_zone_file = (
        Path.cwd().joinpath(conf.input_data_path).joinpath(f"{file_name}.feather")
    )
    f_zone_df = pd.read_feather(input_f_zone_file).sort_values(by="frame")

    input_lyrics_file = (
        Path.cwd().joinpath(conf.lyrics_input_path).joinpath(f"{file_name}.feather")
    )
    input_lyrics_df = pd.read_feather(input_lyrics_file)

    output_file_path = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(f"{file_name}.feather")
    )

    result_df = pd.DataFrame()

    for index, row in input_lyrics_df.iterrows():
        # get a slice from person box df based on the current start-time and end-time in this row of the lyrics df
        f_zone_df_slice = f_zone_df.loc[
            (f_zone_df["frame"] >= row["start_time"])
            & (f_zone_df["frame"] <= row["end_time"])
        ]
        # get sub slices as per IOU from the above slice
        list_of_cut_points = get_cut_points(f_zone_df_slice)
        list_of_split_dfs = split_dfs(f_zone_df_slice, list_of_cut_points)

        # do union of all the sub slices
        for df in list_of_split_dfs:
            x1, y1, x3, y3 = do_union(df)
            row = {
                "start_time": df["frame"].min(),
                "end_time": df["frame"].max(),
                "text": row["text"],
                "font_type": row["font_type"],
                "x1": x1,
                "x3": x3,
                "y1": y1,
                "y3": y3,
            }

            result_df = result_df.append(row, ignore_index=True)

    result_df[["x1", "y1", "x3", "y3"]] = result_df[["x1", "y1", "x3", "y3"]].astype(
        int
    )

    result_df.to_feather(output_file_path)

    return True


if __name__ == "__main__":
    config = Config(
        input_data_path="../data/detected_persons_output",
        output_data_path="../data/splitter_output",
        lyrics_input_path="../data/pre_processed_output",
    )
    config.set_run_id(run_id="d9868ecd-29ec-4cae-8928-f0e027cc56d0")

    split(conf=config)
