from pathlib import Path
import pandas as pd
from configs.make_config import Config


def split(conf: Config) -> bool:
	"""This function must iterate in the increasing order of time"""
	# read feather file : 8a24e911-cb28-4811-8797-7c3ad98032dc
	file_name = "824170fd-b87b-4acc-9e57-82d0e618666b"
	# input_f_zone_file = Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.feather")
	input_f_zone_file = Path.cwd().joinpath(conf.input_data_path).joinpath(f"{file_name}.feather")
	f_zone_df = pd.read_feather(input_f_zone_file).sort_values(by='frame')
	input_lyrics_file = Path.cwd().joinpath(conf.lyrics_input_path).joinpath(f"{file_name}.feather")
	input_lyrics_df = pd.read_feather(input_lyrics_file)
	print(f_zone_df)
	print(input_lyrics_df)
	#steps
	# for every line of lyrics
	#   get a slice from f_zone df
	#   split the slices based on IOU.
	#   union the split slices . (define union function and use in lambda)
	# check readme of splitter to understand the output
	# sub splitting function
	return True


if __name__ == "__main__":
	split(Config(input_data_path='/Users/nik/Work/lyrics_overlay/data/detected_persons_output',
				 output_data_path='/Users/nik/Work/lyrics_overlay/data/splitter_output',
				 lyrics_input_path='/Users/nik/Work/lyrics_overlay/data/pre_processed_output'))