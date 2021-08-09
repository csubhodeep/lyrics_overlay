from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from wand.image import Image

from configs.make_config import Config

DEBUG_DRAW = False


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

# TODO: take this hard-coding to config file
COLOR_HUMAN_BOX = (255, 0, 0)
COLOR_TEXT_BOX = (0, 255, 0)
BOX_EDGE_THICKNESS = 2


def overlay(conf: Config):
    """This function does the following:
    1- cap. open the video and initiate video writer
    2- if frame is in current lyrics-time-range den draw rectangle
    3- if frame has already crossed lyrics-time-range. den lyrics index + = 1
    4- write frame

    Args:
        conf:

    Returns:

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
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(
        str(output_video_file),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (frame_width, frame_height),
    )
    lyrics_index = 0
    computation_done_for_one_lyrics_line = False
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
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
                    # TODO : Wrap this in function if it will reduce space below
                    if DEBUG_DRAW:
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

                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    wand_background_image = Image.from_array(img)
                    # BOTTLENECK ######################
                    # This takes around .1 second which is very slow
                    wand_background_image.composite(
                        transparent_image_with_text,
                        left=lyrics_and_boxes_df.loc[lyrics_index, "x1_opti"],
                        top=lyrics_and_boxes_df.loc[lyrics_index, "y1_opti"],
                    )
                    frame = cv2.cvtColor(
                        np.asarray(wand_background_image), cv2.COLOR_RGB2BGR
                    )
                    ################################

                # Write the frame into the file 'output.avi'
                if frame_ts > lyrics_and_boxes_df.loc[lyrics_index, "end_time"]:
                    lyrics_index += 1
                    computation_done_for_one_lyrics_line = False

            out.write(frame)

            # Display the resulting frame
            # cv2.imshow('frame',frame)

        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return True


if __name__ == "__main__":

    config = Config(
        output_data_path="../data/overlayer_output",
        input_data_path="../data/optimizer_output",
        video_input_path="../data/input",
        img_size=416,
        run_id="e299765c-f5d1-412c-bc17-c1cae7a2a9f8",
    )

    overlay(conf=config)
