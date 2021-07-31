from math import ceil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from configs.make_config import Config

FONT_LIB_PATH = Path.cwd().joinpath("post_processor/font_lib")
DEFAULT_FONT_NAME = "yatra_one.otf"


def resize(
    img_shape: Tuple[int, int], old_img_size: int, coords: Tuple[int, int]
) -> Tuple[int, int]:

    if img_shape[1] >= img_shape[0]:  # for landscape frames
        # unitary method - if image has height = 500 and width = 700
        # for 500 height, width = 700 therefore, for height = 416, width = (700/500)*416
        width = int(img_shape[1] * old_img_size / img_shape[0])
        height = old_img_size
    else:
        height = int(img_shape[0] * old_img_size / img_shape[1])
        width = old_img_size

    x = int((coords[0] / width) * img_shape[1])
    y = int((coords[1] / height) * img_shape[0])

    return x, y


def resize2(img: np.ndarray, new_res: int) -> np.ndarray:

    if img.shape[1] >= img.shape[0]:
        width = int(img.shape[1] * new_res / img.shape[0])
        height = new_res
    else:
        height = int(img.shape[0] * new_res / img.shape[1])
        width = new_res

    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def draw_text_inside_box(
    image: Image,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    pattern: int,
    font_size: int,
    font_path: Path,
) -> Image:
    image_rgba = image.convert("RGBA")
    text_canvas = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_canvas)
    font = ImageFont.truetype(font_path, font_size)
    # draw.rectangle(((x, y), (x+w, y+h)), fill="black") #only debug purpose
    text_x = int(x + font_size / 2)
    text_y = int(y + font_size / 4)
    shadow_width = 3
    shadowcolor = (128, 128, 128, 50)
    shadow_font = ImageFont.truetype(font_path, font_size + int(shadow_width / 2))
    for i in range(0, len(text), pattern):
        text_line = " ".join(text.split(" ")[i:i + pattern])

        # thin border

        draw.text((text_x - shadow_width, text_y), text_line, font=shadow_font, fill=shadowcolor)
        draw.text((text_x + shadow_width, text_y), text_line, font=shadow_font, fill=shadowcolor)
        draw.text((text_x, text_y - shadow_width), text_line, font=shadow_font, fill=shadowcolor)
        draw.text((text_x, text_y + shadow_width), text_line, font=shadow_font, fill=shadowcolor)
        # main text
        draw.text((text_x, text_y), text_line, font=font)

        text_y = text_y + font_size
    combined_image = Image.alpha_composite(image_rgba, text_canvas)
    return combined_image


# it takes x,y , w and h of resized text box (resized according to original image)
def find_font_size_and_pattern(x: int, y: int, w: int, h: int, text: str):
    pattern = int(w / h) + 1
    if pattern < 2:
        pattern = 2
    elif pattern > 5:
        pattern = 5
    max_width = 0
    num_lines = ceil(len(text.split(" ")) / pattern)
    for i in range(0, len(text), pattern):
        length = len(" ".join(text.split(" ")[i : i + pattern]))
        if length > max_width:
            max_width = length
    max_width += 2
    font_size_init = int(h / (num_lines + 1))
    for size in range(font_size_init, int(font_size_init / 4), -1):
        if (size / 2) * max_width < w:
            return size, pattern
    # this is default font size and pattern
    # this should either come from config or some logic
    print("Default font size and pattern used, fix this in future")
    return 10, 2


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

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # frame = resize2(frame, conf.img_size)
            frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            if lyrics_index < len(lyrics_and_boxes_df):
                if (
                    lyrics_and_boxes_df.loc[lyrics_index, "start_time"]
                    <= frame_ts
                    <= lyrics_and_boxes_df.loc[lyrics_index, "end_time"]
                ):
                    first_diag_coord = (
                        lyrics_and_boxes_df.loc[lyrics_index, "x1"],
                        lyrics_and_boxes_df.loc[lyrics_index, "y1"],
                    )
                    second_diag_coord = (
                        lyrics_and_boxes_df.loc[lyrics_index, "x3"],
                        lyrics_and_boxes_df.loc[lyrics_index, "y3"],
                    )
                    first_diag_coord_opti = (
                        lyrics_and_boxes_df.loc[lyrics_index, "x1_opti"],
                        lyrics_and_boxes_df.loc[lyrics_index, "y1_opti"],
                    )
                    second_diag_coord_opti = (
                        lyrics_and_boxes_df.loc[lyrics_index, "x3_opti"],
                        lyrics_and_boxes_df.loc[lyrics_index, "y3_opti"],
                    )
                    color = (255, 0, 0)
                    color_opti = (0, 255, 0)
                    thickness = 2
                    # # TODO: inverse transform the boxes to big resolution before making rectangle
                    start_point = resize(
                        img_shape=frame.shape,
                        old_img_size=conf.img_size,
                        coords=first_diag_coord,
                    )
                    end_point = resize(
                        img_shape=frame.shape,
                        old_img_size=conf.img_size,
                        coords=second_diag_coord,
                    )
                    start_point_opti = resize(
                        img_shape=frame.shape,
                        old_img_size=conf.img_size,
                        coords=first_diag_coord_opti,
                    )
                    end_point_opti = resize(
                        img_shape=frame.shape,
                        old_img_size=conf.img_size,
                        coords=second_diag_coord_opti,
                    )
                    # start_point = first_diag_coord
                    # end_point = second_diag_coord
                    # start_point_opti = first_diag_coord_opti
                    # end_point_opti = second_diag_coord_opti
                    frame = cv2.rectangle(
                        frame, start_point, end_point, color, thickness
                    )

                    frame = cv2.rectangle(
                        frame, start_point_opti, end_point_opti, color_opti, thickness
                    )
                    # You may need to convert the color.
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    text_box_x1 = start_point_opti[0]
                    text_box_y1 = start_point_opti[1]
                    text_box_width = abs(start_point_opti[0] - end_point_opti[0])
                    text_box_height = abs(start_point_opti[1] - end_point_opti[1])
                    lyrics_text = lyrics_and_boxes_df.loc[lyrics_index, "text"]
                    # calculate font size and pattern here from resize box
                    size, pattern = find_font_size_and_pattern(
                        text_box_x1,
                        text_box_y1,
                        text_box_width,
                        text_box_height,
                        lyrics_text,
                    )
                    drawn_pil_img = draw_text_inside_box(
                        image=Image.fromarray(img),
                        x=text_box_x1,
                        y=text_box_y1,
                        w=text_box_width,
                        h=text_box_height,
                        text=lyrics_text,
                        font_path=FONT_LIB_PATH.joinpath(DEFAULT_FONT_NAME),
                        font_size=size,  # not using size from optimizer
                        pattern=pattern,  # not using pattern from optimizer
                    )

                    frame = cv2.cvtColor(np.asarray(drawn_pil_img), cv2.COLOR_RGB2BGR)

                # Write the frame into the file 'output.avi'
                if frame_ts > lyrics_and_boxes_df.loc[lyrics_index, "end_time"]:
                    lyrics_index += 1

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
        output_data_path="../data/final_output",
        input_data_path="../data/optimizer_output",
        video_input_path="../data/input",
        img_size=416,
        run_id="a1945f8a-6fbf-4686-a1fe-486fcfed1590",
    )

    overlay(conf=config)
