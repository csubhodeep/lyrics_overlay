from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from configs.make_config import Config


def resize(img_shape: Tuple[int, int], old_img_size: int, coords: Tuple[int, int]) -> Tuple[int, int]:

	if img_shape[1] >= img_shape[0]:
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


def overlay(conf: Config):
	# TODO: the following steps
	"""
	1- cap. open the video and initiate video writer
	2- if frame is in current lyrics-time-range den draw rectangle
	3- if frame has already crossed lyrics-time-range. den lyrics index + = 1
	4- write frame
	"""
	file_name = conf.run_id
	lyrics_boxes_file = Path.cwd().joinpath(conf.input_data_path).joinpath(f"{file_name}.feather")
	lyrics_and_boxes_df = pd.read_feather(lyrics_boxes_file).sort_values(by='start_time')
	
	input_video_file_name = Path.cwd().joinpath(conf.video_input_path).joinpath(f"{file_name}.mp4")

	output_video_file = Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.avi")

	# Create a VideoCapture object
	cap = cv2.VideoCapture(str(input_video_file_name))

	# Check if camera opened successfully
	if (cap.isOpened() == False):
		print("Unable to read camera feed")

	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	fps = cap.get(cv2.CAP_PROP_FPS)

	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter(str(output_video_file), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
	lyrics_index = 0

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			# frame = resize2(frame, conf.img_size)
			frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
			if lyrics_index < len(lyrics_and_boxes_df):
				if lyrics_and_boxes_df.loc[lyrics_index, 'start_time'] <= frame_ts <= lyrics_and_boxes_df.loc[lyrics_index, 'end_time']:
					first_diag_coord = (lyrics_and_boxes_df.loc[lyrics_index, 'x1'], lyrics_and_boxes_df.loc[lyrics_index, 'y1'])
					second_diag_coord = (lyrics_and_boxes_df.loc[lyrics_index, 'x3'], lyrics_and_boxes_df.loc[lyrics_index, 'y3'])
					first_diag_coord_opti = (lyrics_and_boxes_df.loc[lyrics_index, 'x1_opti'], lyrics_and_boxes_df.loc[lyrics_index, 'y1_opti'])
					second_diag_coord_opti = (lyrics_and_boxes_df.loc[lyrics_index, 'x3_opti'], lyrics_and_boxes_df.loc[lyrics_index, 'y3_opti'])
					color = (255, 0, 0)
					color_opti = (0, 255, 0)
					thickness = 2
					# # TODO: inverse transform the boxes to big resolution before making rectangle
					start_point = resize(img_shape=frame.shape, old_img_size=conf.img_size, coords=first_diag_coord)
					end_point = resize(img_shape=frame.shape, old_img_size=conf.img_size, coords=second_diag_coord)
					start_point_opti = resize(img_shape=frame.shape, old_img_size=conf.img_size, coords=first_diag_coord_opti)
					end_point_opti = resize(img_shape=frame.shape, old_img_size=conf.img_size, coords=second_diag_coord_opti)
					# start_point = first_diag_coord
					# end_point = second_diag_coord
					# start_point_opti = first_diag_coord_opti
					# end_point_opti = second_diag_coord_opti
					frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
					frame = cv2.rectangle(frame, start_point_opti, end_point_opti, color_opti, thickness)
				# Write the frame into the file 'output.avi'
				if frame_ts > lyrics_and_boxes_df.loc[lyrics_index, 'end_time']:
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

	config = Config(output_data_path="../data/final_output",
					input_data_path="../data/optimizer_output",
					video_input_path="../data/input",
					img_size=416)
	config.set_run_id(run_id="c17e21ec-ba7f-4e11-925b-a8d57fe240d9")

	overlay(conf=config)
