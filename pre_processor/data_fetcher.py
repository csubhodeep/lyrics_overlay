
from pathlib import Path
from shutil import copy
from configs.make_config import Config


def fetch_data(conf: Config) -> bool:
	"""This function is responsible to get/pull/download data from a certain location
	to the machine where the pipeline is running"""

	output_file_path_video = Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.mp4")
	output_file_path_lyrics = Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.csv")

	assert Path.cwd().joinpath(conf.output_data_path).exists()

	# # FIXME: right now we are copying the same file to the same location to emulate the same behaviour
	input_file_path_video = Path.cwd().joinpath("oh_oh_jaane_jaana.mp4")
	input_file_path_lyrics = Path.cwd().joinpath("oh_oh_jaane_jaana_lyrics.csv")
	copy(src=input_file_path_video, dst=output_file_path_video)
	copy(src=input_file_path_lyrics, dst=output_file_path_lyrics)

	return True


if __name__ == "__main__":
	fetch_data(conf=Config(output_data_path="./data/input",
						   run_id="asdsa132"))
