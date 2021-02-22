
from pathlib import Path

import cv2
from datetime import datetime
import numpy as np
import pandas as pd

from configs.make_config import Config


def get_milliseconds(time: str) -> int:

	time_object = datetime.strptime(time, "%M:%S.%f")

	return int(time_object.minute*60*1000 + time_object.second*1000 + time_object.microsecond/1000)


def process_lyrics(lyrics: pd.DataFrame) -> pd.DataFrame:

	lyrics['start_time'] = lyrics['start_time'].apply(get_milliseconds)
	lyrics['end_time'] = lyrics['end_time'].apply(get_milliseconds)

	lyrics.sort_values(by='start_time', inplace=True)

	return lyrics


def resize(img: np.ndarray, new_res: int) -> np.ndarray:
	width = int(img.shape[1] * new_res / img.shape[0])
	height = int(img.shape[0] / new_res)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
	return resized


def sample(conf: Config) -> bool:

	input_video_file_name = f"{conf.run_id}.mp4"
	input_lyrics_file_name = f"{conf.run_id}.csv"
	input_video_path = Path.cwd().joinpath(conf.input_data_path).joinpath(input_video_file_name)
	input_lyrics_path = Path.cwd().joinpath(conf.input_data_path).joinpath(input_lyrics_file_name)

	output_folder_path = Path.cwd().joinpath(conf.output_data_path).joinpath(conf.run_id)
	output_folder_path.mkdir(exist_ok=True)

	raw_lyrics_df = pd.read_csv(input_lyrics_path)

	lyrics_df = process_lyrics(raw_lyrics_df)

	cap = cv2.VideoCapture(str(input_video_path))


	# TODO: for each row in the lyrics file
	# check if the timestamp of the current frame falls between start_time and end_time of the lyrics
	# if yes then from the start time sample with the FPS set in the config
	i = 0
	while (cap.isOpened()):
		if i < lyrics_df.shape[0]:
			frame_exists, curr_frame = cap.read()
			if frame_exists:
				frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
				if frame_ts >= lyrics_df.loc[i, 'start_time']:
					if frame_ts <= lyrics_df.loc[i, 'end_time']:
						if frame_ts % conf.sampling_fps == 0:
							output_file_path = output_folder_path.joinpath(f"{frame_ts}.npy")
							with open(str(output_file_path), 'wb') as f:
								np.save(f, resize(curr_frame, conf.min_output_frame_dim))
						else:
							pass
					else:
						i = i + 1
				else:
					pass
			else:
				break
		else:
			break

	cap.release()

	return True