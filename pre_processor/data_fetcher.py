from pathlib import Path
from shutil import copy

from configs.make_config import Config


def download_data(src: Path, dest: Path) -> None:

    # # FIXME: right now we are copying a file from local location to emulate the same behaviour
    copy(src=src, dst=dest)


def fetch_data(conf: Config) -> bool:
    """This function is responsible to get/pull/download data from a certain location
        to the machine where the pipeline is running"""

    output_file_path_video = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.mp4")
    )
    output_file_path_lyrics = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.csv")
    )

    assert Path.cwd().joinpath(conf.output_data_path).exists()

    # TODO: this going to change in future
    input_file_path_video = (
        Path.cwd()
        .joinpath(conf.input_data_path)
        .joinpath("test_cases")
        .joinpath(conf.filename)
    )

    lyrics_file_name = conf.filename.split(".")[0]

    input_file_path_lyrics = (
        Path.cwd().joinpath("test_cases").joinpath(f"{lyrics_file_name}.csv")
    )

    download_data(src=input_file_path_lyrics, dest=output_file_path_lyrics)
    download_data(src=input_file_path_video, dest=output_file_path_video)

    return True


if __name__ == "__main__":
    fetch_data(
        conf=Config(
            output_data_path="./data/input", input_data_path="./data", run_id="asdsa132"
        )
    )
