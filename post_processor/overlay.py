"""
VERY IMPORTANT
if we got landscape image:
like h500,w1000
den we converted it to :
h416,w 832     (so width is not 736)

if we got portrait image h1000, w=500
den we converted it to
h=832,w=416
"""
import os
import time
from functools import partial
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import pandas as pd
from wand.image import Image

from configs.make_config import Config


# TODO: take this hard-coding to config file
COLOR_HUMAN_BOX = (255, 0, 0)
COLOR_TEXT_BOX = (0, 255, 0)
BOX_EDGE_THICKNESS = 2

if os.getenv("ENVIRONMENT") == "test":
    DEBUG = True
else:
    DEBUG = False

q1: Queue = Queue()
q2: Queue = Queue()

# from collections import UserList
# from typing import Any
# from typing import Tuple
# from typing import Union
# class CustomQueue(UserList):
#     def put(self, item: Tuple[Union[int, float], Any]):
#         self.data.append(item)
#         self.data.sort(key=lambda x: x[0])
#
#     def get(self):
#         return self.data.pop(0)
#
#     def empty(self) -> bool:
#         return len(self.data) == 0
#
#     def task_done(self):
#         ...
#
#
# q2 = CustomQueue()


def draw_boxes(frame, lyrics_and_boxes_df: pd.DataFrame, lyrics_index: int):
    frame = cv2.rectangle(
        frame,
        (
            lyrics_and_boxes_df.loc[lyrics_index, "x1"],
            lyrics_and_boxes_df.loc[lyrics_index, "y1"],
        ),
        (
            lyrics_and_boxes_df.loc[lyrics_index, "x3"],
            lyrics_and_boxes_df.loc[lyrics_index, "y3"],
        ),
        COLOR_HUMAN_BOX,
        BOX_EDGE_THICKNESS,
    )

    frame = cv2.rectangle(
        frame,
        (
            lyrics_and_boxes_df.loc[lyrics_index, "x1_opti"],
            lyrics_and_boxes_df.loc[lyrics_index, "y1_opti"],
        ),
        (
            lyrics_and_boxes_df.loc[lyrics_index, "x3_opti"],
            lyrics_and_boxes_df.loc[lyrics_index, "y3_opti"],
        ),
        COLOR_TEXT_BOX,
        BOX_EDGE_THICKNESS,
    )

    return frame


def write(out):
    # FIXME: don't know what would happen q1 and q2 both are empty - could happen if only the overlay is taking time to process the last frame
    while not (q2.empty() and q1.empty()):
        out.write(q2.get()[1])
        q2.task_done()
    out.release()


def compose(frame, lyrics_and_boxes_df, lyrics_index, transparent_image_with_text):
    if DEBUG:
        frame = draw_boxes(frame, lyrics_and_boxes_df, lyrics_index)

    wand_background_image = Image.from_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # BOTTLENECK ######################
    # This takes around .1 second which is very slow
    wand_background_image.composite(
        transparent_image_with_text,
        left=lyrics_and_boxes_df.loc[lyrics_index, "x1_opti"],
        top=lyrics_and_boxes_df.loc[lyrics_index, "y1_opti"],
    )
    ################################
    return cv2.cvtColor(np.asarray(wand_background_image), cv2.COLOR_RGB2BGR)


def compose2(job, lyrics_and_boxes_df, lyrics_index, transparent_image_with_text_blob):
    frame = job["frame"]

    if False:
        frame = draw_boxes(frame, lyrics_and_boxes_df, lyrics_index)

    print("here1", job["frame_ts"])
    transparent_image_with_text = Image(blob=transparent_image_with_text_blob)
    print("here2", job["frame_ts"])

    wand_background_image = Image.from_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # BOTTLENECK ######################
    # This takes around .1 second which is very slow
    wand_background_image.composite(
        transparent_image_with_text,
        left=lyrics_and_boxes_df.loc[lyrics_index, "x1_opti"],
        top=lyrics_and_boxes_df.loc[lyrics_index, "y1_opti"],
    )
    ################################
    return (
        job["frame_ts"],
        cv2.cvtColor(np.asarray(wand_background_image), cv2.COLOR_RGB2BGR),
    )


def parallel_compose(
    jobs, lyrics_and_boxes_df, lyrics_index, transparent_image_with_text_blob
):
    batch_size = 7

    all_results = []

    n_batches = int(ceil(len(jobs) / batch_size))

    for i in range(n_batches):
        batch = jobs[i : i + batch_size]
        with Pool() as p:
            results = p.map(
                partial(
                    compose2,
                    lyrics_and_boxes_df=lyrics_and_boxes_df,
                    lyrics_index=lyrics_index,
                    transparent_image_with_text_blob=transparent_image_with_text_blob,
                ),
                batch,
            )
        all_results.extend(list(results))

    return all_results


def overlay_lyrics2(lyrics_and_boxes_df, wand_folder_path):
    lyrics_index = 0
    computation_done_for_one_lyrics_line = False
    batch = []
    while not q1.empty():
        frame, frame_ts = q1.get()
        if lyrics_index < len(lyrics_and_boxes_df):
            if (
                lyrics_and_boxes_df.loc[lyrics_index, "start_time"]
                <= frame_ts
                <= lyrics_and_boxes_df.loc[lyrics_index, "end_time"]
            ):
                if not computation_done_for_one_lyrics_line:
                    transparent_image_with_text_blob = Image(
                        filename=str(
                            wand_folder_path.joinpath(
                                f"{lyrics_and_boxes_df.loc[lyrics_index, 'start_time']}.png"
                            )
                        )
                    ).make_blob()

                    computation_done_for_one_lyrics_line = True

                batch.append({"frame_ts": frame_ts, "frame": frame})
            elif frame_ts > lyrics_and_boxes_df.loc[lyrics_index, "end_time"]:
                print(len(batch))
                results = parallel_compose(
                    batch,
                    lyrics_and_boxes_df,
                    lyrics_index,
                    transparent_image_with_text_blob,
                )

                print(lyrics_index)
                for res in results:
                    q2.put((res[0], res[1]))

                lyrics_index += 1
                computation_done_for_one_lyrics_line = False
                batch = []
            else:
                q2.put((frame_ts, frame))
        else:
            q2.put((frame_ts, frame))
        q1.task_done()


def overlay_lyrics(lyrics_and_boxes_df, wand_folder_path):
    lyrics_index = 0
    computation_done_for_one_lyrics_line = False
    while not q1.empty():
        frame, frame_ts = q1.get()
        if lyrics_index < len(lyrics_and_boxes_df):
            if (
                lyrics_and_boxes_df.loc[lyrics_index, "start_time"]
                <= frame_ts
                <= lyrics_and_boxes_df.loc[lyrics_index, "end_time"]
            ):
                if not computation_done_for_one_lyrics_line:
                    transparent_image_with_text = Image(
                        filename=str(
                            wand_folder_path.joinpath(
                                f"{lyrics_and_boxes_df.loc[lyrics_index, 'start_time']}.png"
                            )
                        )
                    )

                    computation_done_for_one_lyrics_line = True

                frame = compose(
                    frame,
                    lyrics_and_boxes_df,
                    lyrics_index,
                    transparent_image_with_text,
                )
            elif frame_ts > lyrics_and_boxes_df.loc[lyrics_index, "end_time"]:
                lyrics_index += 1
                computation_done_for_one_lyrics_line = False

        q2.put((frame_ts, frame))
        q1.task_done()


def read(cap):
    flg = cap.isOpened()
    while flg:
        ret, frame = cap.read()
        if ret:
            frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            q1.put(item=(frame, frame_ts))
        flg = not q1.empty()

    cap.release()


def overlay(conf: Config):
    """This function does the following:
    1- cap. open the video and initiate video writer
    2- if frame is in current lyrics-time-range den draw rectangle
    3- if frame has already crossed lyrics-time-range. den lyrics index + = 1
    4- write frame

    """
    file_name = conf.run_id
    lyrics_boxes_file = (
        Path.cwd().joinpath(conf.input_data_path).joinpath(f"{file_name}.feather")
    )
    lyrics_and_boxes_df = pd.read_feather(lyrics_boxes_file).sort_values(
        by="start_time"
    )

    wand_folder_path = conf.input_data_path.joinpath(f"{conf.run_id}")

    input_video_file_name = (
        Path.cwd().joinpath(conf.video_input_path).joinpath(f"{file_name}.mp4")
    )

    output_video_file = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.avi")
    )

    # Create a VideoCapture object
    cap = cv2.VideoCapture(str(input_video_file_name))

    # Check if camera opened successfully
    if not cap.isOpened():
        raise Exception("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    # Define the codec and create VideoWriter object.The output is stored in a '.avi' file.
    out = cv2.VideoWriter(
        str(output_video_file),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4))),
    )
    th_read = Thread(target=read, args=(cap,), daemon=True)
    th_overlay = Thread(
        target=overlay_lyrics, args=(lyrics_and_boxes_df, wand_folder_path)
    )
    th_write = Thread(target=write, args=(out,))

    th_read.start()
    time.sleep(0.5)
    th_overlay.start()
    th_write.start()

    th_read.join()
    th_overlay.join()
    th_write.join()

    # Closes all the frames
    cv2.destroyAllWindows()

    return True


if __name__ == "__main__":

    config = Config(
        output_data_path="../data/overlayer_output",
        input_data_path="../data/optimizer_output",
        video_input_path="../data/input",
        img_size=416,
        run_id="fdac852c-94b4-4ab5-bccb-566cd90c7e64",
    )

    ts = time.time()
    overlay(conf=config)
    print(time.time() - ts)
