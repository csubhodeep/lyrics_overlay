
from pathlib import Path
from typing import Union

class Config:
	__slots__ = (
		'_input_data_path',
		'_output_data_path'
	)

	def __init__(self,
				output_data_path: Union[str, Path],
				input_data_path: Union[str, Path] = ""):
		self._input_data_path = Path(input_data_path) if isinstance(input_data_path, str) else input_data_path
		self._output_data_path = Path(output_data_path) if isinstance(output_data_path, str) else output_data_path

	@property
	def input_data_path(self) -> Union[str, Path]:
		return self._input_data_path

	@property
	def output_data_path(self) -> Union[str, Path]:
		return self._output_data_path

	def set_input_data_path(self, new_path: Union[str, Path]) -> None:
		self._input_data_path = Path(new_path) if isinstance(new_path, str) else new_path

	def set_output_data_path(self, new_path: Union[str, Path]) -> None:
		self._output_data_path = Path(new_path) if isinstance(new_path, str) else new_path


def get_config(path_to_config: Union[Path, str]) -> Config:

	## TODO: take configs from json file
	config = Config(output_data_path="")



	return config