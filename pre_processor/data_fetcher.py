
import json
import os
from pathlib import Path

from configs.make_config import Config


def fetch_data(conf: Config) -> bool:

	# doing random stuff
	os.system(f"cp {conf.input_data_path}/girls_like_you_small.mp4 {conf.output_data_path}/{conf.run_id}.mp4")

	return True


if __name__ == "__main__":
	fetch_data(conf=Config(output_data_path="./data/input",
						   input_data_path="./data/input"))