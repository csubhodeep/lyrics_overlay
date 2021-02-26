from pathlib import Path
import cv2
import numpy as np
from configs.make_config import Config


def resize(img: np.ndarray, new_res: int) -> np.ndarray:

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
	
	input_video_file_name = f"{file_name}.mp4"


	# Create a VideoCapture object
	cap = cv2.VideoCapture(input_video_file_name)

	# Check if camera opened successfully
	if (cap.isOpened() == False):
		print("Unable to read camera feed")

	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	fps = cap.get(cv2.CAP_PROP_FPS)


	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

	while(True):
		ret, frame = cap.read()
		lyrics_index = 0
		if ret == True:
			frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
			if frame_ts > lyrics_and_boxes_df.loc[lyrics_index, 'end_time']:
				lyrics_index += 1
			if frame_ts >= lyrics_and_boxes_df.loc[lyrics_index, 'start_time']:
				start_point = (lyrics_and_boxes_df.loc[lyrics_index, 'x1'], lyrics_and_boxes_df.loc[lyrics_index, 'y1'])
				end_point = (lyrics_and_boxes_df.loc[lyrics_index, 'x3'], lyrics_and_boxes_df.loc[lyrics_index, 'y3'])
				color = (255, 0, 0)
				thickness = 2
				# todo inverse transform the boxes to big resolution before making rectangle
				frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
			# Write the frame into the file 'output.avi'
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
					input_data_path="../data/optimizer_output")
	config.set_run_id(run_id="asdsadsadsad")

	overlay(conf=config)
