from pathlib import Path
import pandas as pd
from configs.make_config import Config


def split(conf: Config) -> bool:
	"""This function must iterate in the increasing order of time"""
	file_name = conf.run_id
	# file_name = "824170fd-b87b-4acc-9e57-82d0e618666b"
	input_f_zone_file = Path.cwd().joinpath(conf.input_data_path).joinpath(f"{file_name}.feather")
	f_zone_df = pd.read_feather(input_f_zone_file).sort_values(by='frame')
	input_lyrics_file = Path.cwd().joinpath(conf.lyrics_input_path).joinpath(f"{file_name}.feather")
	input_lyrics_df = pd.read_feather(input_lyrics_file)

	result_df = pd.DataFrame()
	lyrics_index = 0
	x1 = 500  # todo max int
	y1 = 500  # todo max int
	x3 = 0
	y3 = 0
	for index, row in f_zone_df.iterrows():
		if row['frame'] < input_lyrics_df.loc[lyrics_index, 'start_time']:
			continue
		if row['frame'] > input_lyrics_df.loc[lyrics_index, 'end_time']:
			row = {
				"x1": x1,
				"x3": x3,
				"y1": y1,
				"y3": y3
			}
			result_df = result_df.append(row, ignore_index=True)
			lyrics_index += 1
			x1 = 500  # todo max int
			y1 = 500  # todo max int
			x3 = 0
			y3 = 0
			continue
		# todo Calculate IOU here and compare with last IOU if its a drastic change than cut here and add a row in df
		x1 = min(x1, row['x1'])
		y1 = min(y1, row['y1'])
		x3 = max(x3, row['x3'])
		y3 = max(y3, row['y3'])

	# insert one-last row that is missed in the loop
	if x3:  # todo think about better check
		row = {
			"x1": x1,
			"x3": x3,
			"y1": y1,
			"y3": y3
		}
		result_df = result_df.append(row, ignore_index=True)

	result_df = input_lyrics_df.join(result_df, lsuffix='_caller', rsuffix='_other')
	return result_df


if __name__ == "__main__":
	split(Config(input_data_path='/Users/nik/Work/lyrics_overlay/data/detected_persons_output',
				 output_data_path='/Users/nik/Work/lyrics_overlay/data/splitter_output',
				 lyrics_input_path='/Users/nik/Work/lyrics_overlay/data/pre_processed_output'))