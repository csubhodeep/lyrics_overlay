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
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import pandas as pd

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


def draw_boxes(frame, x1, y1, x3, y3, x1_opti, y1_opti, x3_opti, y3_opti):
    frame = cv2.rectangle(
        frame, (x1, y1), (x3, y3), COLOR_HUMAN_BOX, BOX_EDGE_THICKNESS
    )

    frame = cv2.rectangle(
        frame,
        (x1_opti, y1_opti),
        (x3_opti, y3_opti),
        COLOR_TEXT_BOX,
        BOX_EDGE_THICKNESS,
    )

    return frame


def write(out):
    # FIXME: don't know what would happen q1 and q2 both are empty - could happen if only the overlay is taking time to process the last frame
    while not (q2.empty() and q1.empty()):
        out.write(q2.get())
        q2.task_done()
    out.release()


def overlay_transparent(background, x, y, overlay):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype)
                * 255,
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / np.max(overlay[..., 3:])

    background[y : y + h, x : x + w] = (1.0 - mask) * background[
        y : y + h, x : x + w
    ] + mask * overlay_image

    return background


def compose(
    frame,
    transparent_image_with_text,
    x1_opti,
    y1_opti,
    x3_opti,
    y3_opti,
    x1,
    y1,
    x3,
    y3,
):

    if DEBUG:
        frame = draw_boxes(frame, x1, y1, x3, y3, x1_opti, y1_opti, x3_opti, y3_opti)

    return overlay_transparent(
        background=frame, x=x1_opti, y=y1_opti, overlay=transparent_image_with_text
    )


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
                    transparent_image_with_text = cv2.imread(
                        filename=str(
                            wand_folder_path.joinpath(
                                f"{lyrics_and_boxes_df.loc[lyrics_index, 'start_time']}.png"
                            )
                        ),
                        flags=cv2.IMREAD_UNCHANGED,
                    )

                    computation_done_for_one_lyrics_line = True

                frame = compose(
                    frame=frame,
                    x1_opti=lyrics_and_boxes_df.loc[lyrics_index, "x1_opti"],
                    y1_opti=lyrics_and_boxes_df.loc[lyrics_index, "y1_opti"],
                    x1=lyrics_and_boxes_df.loc[lyrics_index, "x1"],
                    y1=lyrics_and_boxes_df.loc[lyrics_index, "y1"],
                    x3_opti=lyrics_and_boxes_df.loc[lyrics_index, "x3_opti"],
                    y3_opti=lyrics_and_boxes_df.loc[lyrics_index, "y3_opti"],
                    x3=lyrics_and_boxes_df.loc[lyrics_index, "x3"],
                    y3=lyrics_and_boxes_df.loc[lyrics_index, "y3"],
                    transparent_image_with_text=transparent_image_with_text,
                )

            elif frame_ts > lyrics_and_boxes_df.loc[lyrics_index, "end_time"]:
                # using multiprocessing to process a batch of frames for each line of lyrics
                # leads to more time consumption (almost double) due to the need of sorting
                # the results according to their timestamps

                lyrics_index += 1
                computation_done_for_one_lyrics_line = False

        q2.put(frame)
        q1.task_done()


def read(cap):
    flg = cap.isOpened()
    while flg:
        ret, frame = cap.read()
        if ret:
            q1.put(item=(frame, cap.get(cv2.CAP_PROP_POS_MSEC)))
        # a weird hack to close this Thread once the next thread has consumed all the tasks
        # produced by this thread
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
        run_id="75e1e27a-630b-4db2-bbc2-82cf9b5c5ee5",
    )

    overlay(conf=config)
