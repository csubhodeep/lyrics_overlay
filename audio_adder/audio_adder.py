from pathlib import Path
import os
from configs.make_config import Config
import moviepy.editor as mp


def add_audio(conf: Config) -> bool:
    file_name = conf.run_id
    video_with_audio_path = Path.cwd().joinpath(conf.video_input_path).joinpath(f"{file_name}.mp4")
    video_without_audio_path = Path.cwd().joinpath(conf.input_data_path).joinpath(f"{file_name}.avi")

    my_clip_with_audio = mp.VideoFileClip(str(video_with_audio_path))
    my_clip_wo_audio = mp.VideoFileClip(str(video_without_audio_path))

    output_audio_path = Path.cwd().joinpath(conf.output_data_path).joinpath(f"{file_name}.mp3")
    my_clip_with_audio.audio.write_audiofile(output_audio_path)
    audioclip = mp.AudioFileClip(str(output_audio_path))

    videoclip = my_clip_wo_audio.set_audio(audioclip)

    output_video_path = Path.cwd().joinpath(conf.output_data_path).joinpath(f"{file_name}.mp4")
    videoclip.write_videofile(str(output_video_path))

    return True


if __name__ == "__main__":
    config = Config(
        input_data_path="../data/final_output",
        output_data_path="../data/final_output",
        video_input_path="../data/input",
    )
    config.set_run_id(run_id="d60576eb-cd1d-4842-83c1-41e6f02b593e")

    add_audio(conf=config)
