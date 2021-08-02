import cv2
import mediapipe as mp
from pathlib import Path
import pandas as pd
from configs.make_config import Config
import numpy as np
from typing import Dict, List, Tuple, Union
import math

PRESENCE_THRESHOLD = 0.5
RGB_CHANNELS = 3
RED_COLOR = (0, 0, 255)
VISIBILITY_THRESHOLD = 0.5


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int,
                                     image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def person_bounding_box(frame, normalized_body_keypoints_1person):
    image_rows, image_cols, _ = frame.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(normalized_body_keypoints_1person.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    x_min = image_cols
    y_min = image_rows
    x_max = 0
    y_max = 0

    for key in idx_to_coordinates:
        #  13 to 22 are hand and elbow keypoints
        if 13 <= key <= 22:
            continue
        x_min = min(x_min, idx_to_coordinates[key][0])
        x_max = max(x_max, idx_to_coordinates[key][0])
        y_min = min(y_min, idx_to_coordinates[key][1])
        y_max = max(y_max, idx_to_coordinates[key][1])

    person = {"x1": x_min,
              "y1": y_min,
              "x3": x_max,
              "y3": y_max}
    return person


def get_persons(
    frame_info: Tuple[np.ndarray, Path], model,
) -> List[Dict[str, float]]:

    frame = cv2.cvtColor(frame_info[0], cv2.COLOR_BGR2RGB)

    results = model.process(frame)
    if results.pose_landmarks is not None:
        person = person_bounding_box(frame, results.pose_landmarks)
    else:
        person = {"x1": -1,
                  "y1": -1,
                  "x3": -1,
                  "y3": -1}
    person["frame"] = float(frame_info[1].name.rstrip(".npy"))
    return [person]


def detect_persons(conf: Config) -> bool:
    input_frames_path = Path.cwd().joinpath(conf.input_data_path).joinpath(conf.run_id)
    output_file_path = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.feather")
    )
    mp_pose = mp.solutions.pose
    model = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    result_df = pd.DataFrame()

    persons = []
    for item in input_frames_path.iterdir():
        with open(str(item), "rb") as f:
            frame = np.load(f)
        persons.extend(
            get_persons(frame_info=(frame, item), model=model)
             # this should be dict of x1 y1 x3 y3 frame
        )

    for person in persons:
        result_df = result_df.append(person, ignore_index=True)

    result_df.to_feather(output_file_path)

    return True


if __name__ == "__main__":
    config = Config(
        input_data_path="../data/pre_processor_output",
        output_data_path="../data/person_box_detector_output",
        run_id="999c0dfd-3017-4359-a902-3960489f8b48",
        conf_thresh=0.8,
        img_size=416,
        img_height=416,
        img_width=739,
    )

    detect_persons(conf=config)