from pathlib import Path

import moviepy.editor as mp

from configs.make_config import Config


def add_audio(conf: Config) -> bool:
    file_name = conf.run_id
    input_video_with_audio_path = (
        Path.cwd().joinpath(conf.video_input_path).joinpath(f"{file_name}.mp4")
    )
    input_video_without_audio_path = (
        Path.cwd().joinpath(conf.input_data_path).joinpath(f"{file_name}.avi")
    )
    output_video_path = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(f"{file_name}.mp4")
    )

    input_clip_with_audio = mp.VideoFileClip(str(input_video_with_audio_path))
    input_clip_wo_audio = mp.VideoFileClip(str(input_video_without_audio_path))

    output_video_with_audio = input_clip_wo_audio.set_audio(input_clip_with_audio.audio)

    output_video_with_audio.write_videofile(str(output_video_path))

    return True


if __name__ == "__main__":
    config = Config(
        input_data_path="../data/overlayer_output",
        output_data_path="../data/final_output",
        video_input_path="../data/input",
        run_id="1e8e31c5-fa0d-482b-8e40-7c1c1eff769d",
    )

    add_audio(conf=config)
