import cv2
import pandas as pd
from pre_processor.sampler import get_milliseconds
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from string import ascii_letters
import textwrap

def put_text(img,start_point, text, font_size=35, font_path='/home/nikhil/Work/Code/lyrics_overlay/post_processor'
                                                            '/font_lib/typewriter_b.ttf'):
    # You may need to convert the color.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)


    # Open image
    #img = Image.open(fp='background.jpg', mode='r')
    # Load custom font
    font = ImageFont.truetype(font=font_path, size=font_size)
    # Create DrawText object
    draw = ImageDraw.Draw(im=img)
    # Define our text
    text = text
    # Calculate the average length of a single character of our font.
    # Note: this takes into account the specific font and font size.
    avg_char_width = sum(font.getsize(char)[0] for char in ascii_letters) / len(ascii_letters)
    # Translate this average length into a character count
    max_char_count = int(img.size[0] / avg_char_width)
    # Create a wrapped text object using scaled character count
    text = textwrap.fill(text=text, width=max_char_count+7)
    # Add text to the image
    draw.text(xy=start_point, text=text, font=font, fill='#FFFFFF', anchor='mm')

    numpy_image = np.array(img)

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
    # the color is converted from RGB to BGR format
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

def put_text_on_video(video_path, text_path):
    cap = cv2.VideoCapture(video_path)
    raw_lyrics_df = pd.read_csv(text_path)
    current_text_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            if current_time <= get_milliseconds(raw_lyrics_df.iloc[current_text_index]['end_time']) and \
                    current_time >= get_milliseconds(raw_lyrics_df.iloc[current_text_index]['start_time']):
                frame = put_text(frame, (534, 960), raw_lyrics_df.iloc[current_text_index]['text'], )
            elif current_time > get_milliseconds(raw_lyrics_df.iloc[current_text_index]['start_time']):
                if current_text_index < len(raw_lyrics_df)-1:
                    current_text_index += 1
            out.write(frame)
        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    video_path = "/home/nikhil/Work/Code/lyrics_overlay/test_cases/fight_club.mp4"
    text_path = "/home/nikhil/Work/Code/lyrics_overlay/test_cases/fight_club.csv"
    put_text_on_video(video_path, text_path)