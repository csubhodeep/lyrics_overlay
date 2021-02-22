
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from configs.make_config import Config


def sort_lyrics_df(lyrics: pd.DataFrame) -> pd.DataFrame:
	# # TODO: implement sorting according to start-time

	return lyrics


def sample(conf: Config) -> bool:

	input_video_file_name = f"{conf.run_id}.mp4"
	input_lyrics_file_name = f"{conf.run_id}.csv"
	input_video_path = Path.cwd().joinpath(conf.input_data_path).joinpath(input_video_file_name)
	input_lyrics_path = Path.cwd().joinpath(conf.input_data_path).joinpath(input_lyrics_file_name)

	output_folder_path = Path.cwd().joinpath(conf.output_data_path)
	output_folder_path.mkdir(exist_ok=True)

	lyrics_df = pd.read_csv(input_lyrics_path)

	sorted_lyrics_df = sort_lyrics_df(lyrics_df)

	cap = cv2.VideoCapture(str(input_video_path))


	# TODO: for each row in the lyrics file
	# check if the timestamp of the current frame falls between start_time and end_time of the lyrics
	# if yes then from the start time sample with the FPS set in the config
	i = 0
	while (cap.isOpened()):
		frame_exists, curr_frame = cap.read()
		if frame_exists:
			frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
			if frame_ts >= lyrics_df['start_time', i]:
				if frame_ts <= lyrics_df['end_time', i]:
					if frame_ts % conf.sampling_fps == 0:
						output_file_path = output_folder_path.joinpath(f"{frame_ts}.npy")
						with open(str(output_file_path), 'wb') as f:
							np.save(f, curr_frame)
					else:
						pass
				else:
					i = i + 1
			else:
				pass
		else:
			break

	cap.release()

	return True


def test_code():

	input_path = '../girls_like_you_small.mp4'
	ouput_path = '../data/input/xyz.avi'

	# Create a VideoCapture object
	cap = cv2.VideoCapture(input_path)

	fps = cap.get(cv2.CAP_PROP_FPS)

	# Check if camera opened successfully
	if (cap.isOpened() == False):
		print("Unable to read camera feed")

	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))


	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter(ouput_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

	while (True):
		ret, frame = cap.read()

		if ret:

			# Write the frame into the file 'output.avi'
			out.write(frame)

			# Display the resulting frame
			cv2.imshow('frame', frame)

			# Press Q on keyboard to stop recording
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# Break the loop
		else:
			break

	# When everything done, release the video capture and video write objects
	cap.release()
	out.release()

	# Closes all the frames
	cv2.destroyAllWindows()
