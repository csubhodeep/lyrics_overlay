from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Union

import hjson

from pipeline.lib.decorators import make_immutable


class Config:
    """This class creates a portable object with immutable but flexible number of attributes"""

    def __init__(
        self,
        output_data_path: Union[str, Path],
        input_data_path: Optional[Union[str, Path]] = "",
        **kwargs,
    ):
        if input_data_path:
            self.input_data_path = (
                Path(input_data_path)
                if isinstance(input_data_path, str)
                else input_data_path
            )
            assert self.input_data_path.exists(), "Input data path must exist"
        self.output_data_path = (
            Path(output_data_path)
            if isinstance(output_data_path, str)
            else output_data_path
        )

        assert self.output_data_path.exists(), "Output data path must exist"

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @make_immutable(allowed_settable_attributes=())
    def __setattr__(self, key, value):
        """This function ensures immutability of every instance of this class"""
        self.__dict__[key] = value


def get_config(path_to_config: Union[Path, str]) -> Dict[str, Config]:
    with open(str(path_to_config), "r") as f:
        config_dict = hjson.load(f)

    config_collection = {}
    for k, v in config_dict.items():
        config_collection[k] = Config(**v)

    return config_collection


if __name__ == "__main__":
    get_config(path_to_config="config.json")
