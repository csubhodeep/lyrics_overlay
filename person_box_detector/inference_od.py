from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import Compose
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from configs.make_config import Config
from person_box_detector.models import Darknet
from person_box_detector.utils.utils import load_classes
from person_box_detector.utils.utils import non_max_suppression


# Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor


def detect_image(
    img: Image, img_size: int, model: Darknet, conf_thresh: float, nms_thresh: float
):
    # scale and pad image
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = Compose(
        [
            Resize((imh, imw)),
            Pad(
                (
                    max(int((imh - imw) / 2), 0),
                    max(int((imw - imh) / 2), 0),
                    max(int((imh - imw) / 2), 0),
                    max(int((imw - imh) / 2), 0),
                ),
                (128, 128, 128),
            ),
            ToTensor(),
        ]
    )
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thresh, nms_thresh)
    return detections[0]


def post_process_detection(
    detections, classes, pilimg: Image, conf: Config
) -> List[Dict[str, float]]:
    image_height = pilimg.size[1]
    image_width = pilimg.size[0]
    list_of_persons = []
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (conf.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (conf.img_size / max(img.shape))
    unpad_h = conf.img_size - pad_y
    unpad_w = conf.img_size - pad_x
    for x1, y1, x3, y3, cnf, cls_conf, cls_pred in detections:
        cls = classes[int(cls_pred)]
        if cls == "person":
            box_h = int(((y3 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x3 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            x3 = x1 + box_w
            y3 = y1 + box_h
            list_of_persons.append(
                {
                    "x1": np.clip(x1, 0, image_width - 1),
                    "y1": np.clip(y1, 0, image_height - 1),
                    "x3": np.clip(x3, 0, image_width - 1),
                    "y3": np.clip(y3, 0, image_height - 1),
                }
            )

    return list_of_persons


def get_persons(
    frame_info: Tuple[np.ndarray, Path], conf: Config, model: Darknet, classes
) -> List[Dict[str, float]]:

    frame = cv2.cvtColor(frame_info[0], cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(
        pilimg, conf.img_size, model, conf.conf_thresh, conf.nms_thresh
    )
    if detections is not None:
        persons = post_process_detection(detections, classes, pilimg, conf)

        for person in persons:
            person["frame"] = float(frame_info[1].name.rstrip(".npy"))

        return persons
    else:
        return [
            {
                "x1": -1,
                "y1": -1,
                "x3": -1,
                "y3": -1,
                "frame": float(frame_info[1].name.rstrip(".npy")),
            }
        ]


def detect_persons(conf: Config) -> bool:

    input_frames_path = Path.cwd().joinpath(conf.input_data_path).joinpath(conf.run_id)
    output_file_path = (
        Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.feather")
    )

    # Load model and weights
    model = Darknet(conf.config_path, img_size=conf.img_size)
    model.load_weights(conf.weights_path)
    # model.cuda()
    model.eval()
    classes = load_classes(conf.class_path)

    result_df = pd.DataFrame()

    persons = []
    for item in input_frames_path.iterdir():
        with open(str(item), "rb") as f:
            frame = np.load(f)
        persons.extend(
            get_persons(
                frame_info=(frame, item), conf=conf, model=model, classes=classes
            )
        )

    for person in persons:
        result_df = result_df.append(person, ignore_index=True)

    result_df.to_feather(output_file_path)

    return True


if __name__ == "__main__":
    config = Config(
        input_data_path="../data/pre_processor_output",
        output_data_path="../data/person_box_detector_output",
        run_id="bffaf4d5-bdd3-4563-8af1-f90d8b1601aa",
        conf_thresh=0.8,
        nms_thresh=0.4,
        img_size=416,
        img_height=416,
        img_width=739,
        config_path="../person_box_detector/config/yolov3.cfg",
        weights_path="../person_box_detector/config/yolov3.weights",
        class_path="../person_box_detector/config/coco.names",
    )

    detect_persons(conf=config)
