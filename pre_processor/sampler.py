
import json
import os
from pathlib import Path

import cv2


from configs.make_config import Config


def sample(conf: Config) -> bool:

	input_file_name = f"{conf.run_id}.mp4"
	input_path = Path(os.getcwd()).joinpath(conf.input_data_path).joinpath(input_file_name)
	output_file_name = f"{conf.run_id}.json"
	output_file_path = Path(Path(os.getcwd())).joinpath(conf.output_data_path).joinpath(output_file_name)

	cap = cv2.VideoCapture(str(input_path))

	some_output = {}

	while (cap.isOpened()):
		frame_exists, curr_frame = cap.read()
		if frame_exists:
			some_output[str(cap.get(cv2.CAP_PROP_POS_MSEC))] = curr_frame.shape
		else:
			break

	cap.release()

	# save the output
	with open(f"{output_file_path}", 'w') as f:
		json.dump(some_output, f)

	return True


def test_code():

	input_path = '../data/input/girls_like_you_small.mp4'
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


if __name__ == "__main__":
	sample(conf=Config(output_data_path="./data/pre_processed_output",
					   input_data_path="./data/input",
					   run_id="girls_like_you_small.mp4"))