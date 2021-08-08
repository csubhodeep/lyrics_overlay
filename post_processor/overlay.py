import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from configs.make_config import Config

FONT_LIB_PATH = Path(__file__).parent.joinpath("font_lib")
# DEFAULT_FONT_NAME = "Black.otf"
DEBUG_DRAW = True


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


def draw_text_inside_box(
    image: Image,
    x1: int,
    y1: int,
    text: str,
    pattern: int,
    font_size: int,
    font_path: Path,
) -> Image:
    image_rgba = image.convert("RGBA")
    text_canvas = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_canvas)
    font = ImageFont.truetype(str(font_path), font_size)
    # draw.rectangle(((x, y), (x+w, y+h)), fill="black") #only debug purpose
    text_x = int(x1 + font_size / 2)
    text_y = int(y1 + font_size / 4)
    shadow_width = 3
    shadowcolor = (128, 128, 128, 50)
    shadow_font = ImageFont.truetype(str(font_path), font_size + int(shadow_width / 2))
    for i in range(0, len(text), pattern):
        text_line = " ".join(text.split(" ")[i : i + pattern])

        # thin border
        # draw.text(
        #     (text_x - shadow_width, text_y),
        #     text_line,
        #     font=shadow_font,
        #     fill=shadowcolor,
        # )
        # draw.text(
        #     (text_x + shadow_width, text_y),
        #     text_line,
        #     font=shadow_font,
        #     fill=shadowcolor,
        # )
        # draw.text(
        #     (text_x, text_y - shadow_width),
        #     text_line,
        #     font=shadow_font,
        #     fill=shadowcolor,
        # )
        # draw.text(
        #     (text_x, text_y + shadow_width),
        #     text_line,
        #     font=shadow_font,
        #     fill=shadowcolor,
        # )
        draw.text(
            (text_x + shadow_width, text_y + shadow_width),
            text_line,
            font=shadow_font,
            fill=shadowcolor,
        )
        # main text
        draw.text((text_x, text_y), text_line, font=font)

        text_y = text_y + font_size
    combined_image = Image.alpha_composite(image_rgba, text_canvas)
    return combined_image


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
                        # TODO: take this hard-coding to config file
                        DEFAULT_FONT_NAME = random.choice(
                            ["Playlist_Script.otf", "Black.otf", "yatra_one.ttf"]
                        )
                        color = (255, 0, 0)
                        color_opti = (0, 255, 0)
                        thickness = 2
                        computation_done_for_one_lyrics_line = True
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
                            color,
                            thickness,
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
                            color_opti,
                            thickness,
                        )
                    ################
                    # New feature
                    # what if user provides a fix box for lyrics
                    # the following box is for GUL video
                    #####################
                    # text_box_x1 = 130
                    # text_box_y1 = 270
                    # text_box_width = 240
                    # text_box_height = 120
                    ######################
                    # You may need to convert the color.
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # #### debug ## text animation
                    # text_box_x1 = text_box_x1 + random.choice([int(-size/20), int(size/20)])
                    # text_box_y1 = text_box_y1 + random.choice([int(-size/20), int(size/20)])
                    # ###

                    # debug minimum font size
                    # this will sure someday put text outside frame
                    # but we need to solve this
                    ########
                    drawn_pil_img = draw_text_inside_box(
                        image=Image.fromarray(img),
                        x1=lyrics_and_boxes_df.loc[lyrics_index, "x1_opti"],
                        y1=lyrics_and_boxes_df.loc[lyrics_index, "y1_opti"],
                        text=lyrics_and_boxes_df.loc[lyrics_index, "text"],
                        font_path=FONT_LIB_PATH.joinpath(DEFAULT_FONT_NAME),
                        font_size=lyrics_and_boxes_df.loc[
                            lyrics_index, "font_size"
                        ],  # not using size from optimizer
                        pattern=lyrics_and_boxes_df.loc[
                            lyrics_index, "pattern"
                        ],  # not using pattern from optimizer
                    )
                    frame = cv2.cvtColor(np.asarray(drawn_pil_img), cv2.COLOR_RGB2BGR)

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
        run_id="a36f77aa-ae02-40de-9fe1-ea6daa9522be",
    )

    overlay(conf=config)
