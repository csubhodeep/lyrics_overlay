from collections import UserDict
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Union

import json


class Config(UserDict):
	"""This class creates a portable object with immutable but flexible number of attributes"""

	def __init__(self, output_data_path: Union[str, Path], input_data_path: Optional[Union[str, Path]] = "", **kwargs):
		super().__init__()
		# # TODO: assert here if path exists or not
		if input_data_path:
			self._input_data_path = Path(input_data_path) if isinstance(input_data_path, str) else input_data_path
		self._output_data_path = Path(output_data_path) if isinstance(output_data_path, str) else output_data_path

		for k, v in kwargs.items():
			self.__setattr__(k, v)

	def __setattr__(self, key, value):
		"""This function ensures immutability of every instance of this class"""
		if hasattr(self, key):
			raise Exception(f"Attribute - {key} is already set !")

		self.__dict__[key] = value

	@property
	def input_data_path(self) -> Path:
		return self._input_data_path

	@property
	def output_data_path(self) -> Path:
		return self._output_data_path

	@property
	def run_id(self) -> str:
		return self._run_id

	def set_run_id(self, run_id: str):
		self._run_id = run_id

	def set_input_data_path(self, new_path: Union[str, Path]) -> None:
		self._input_data_path = Path(new_path) if isinstance(new_path, str) else new_path


def get_config(path_to_config: Union[Path, str]) -> Dict[str, Config]:
	with open(str(path_to_config), 'r') as f:
		config_dict = json.load(f)

	config_collection = {}
	for k, v in config_dict.items():
		config_collection[k] = Config(**v)

	return config_collection


if __name__ == "__main__":
	get_config(path_to_config='config.json')
