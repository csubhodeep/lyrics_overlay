from pathlib import Path
import json

from configs.make_config import Config


def split(conf: Config) -> bool:
	"""This function must iterate in the increasing order of time"""
	# doing random stuff
	some_output = {}
	file_name = f"{conf.run_id}.json"
	file_path = Path(conf.output_data_path).joinpath(file_name)
	with open(file_path, 'w') as f:
		json.dump(some_output, f)

	return True