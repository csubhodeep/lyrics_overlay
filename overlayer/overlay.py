import json
from pathlib import Path

from configs.make_config import Config

def overlay(conf: Config):
	# doing random stuff
	some_output = {}
	file_name = f"{conf.run_id}.json"
	file_path = Path(conf.output_data_path).joinpath(file_name)
	with open(file_path, 'w') as f:
		json.dump(some_output, f)

	return True


if __name__ == "__main__":
	overlay()