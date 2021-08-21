from functools import partial
from pathlib import Path
from typing import Dict
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
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


def npy_loader(path):
    numpy_image = np.load(path)
    frame = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    return pilimg


class ImageFolderWithPaths(datasets.DatasetFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.samples[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


def get_transformer(img: Image, img_size: int) -> Variable:
    # scale and pad image
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transform = Compose(
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
    return Variable(img_transform(img).float().type(Tensor))


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


def get_only_biggest_person(persons: List[Dict[str, float]]) -> List[Dict[str, float]]:

    return [max(persons, key=lambda x: (x["x3"] - x["x1"]) * (x["y3"] - x["y1"]))]


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

    for inputs, labels, paths in DataLoader(
        ImageFolderWithPaths(
            str(input_frames_path),
            transform=partial(get_transformer, img_size=conf.img_size),
            loader=npy_loader,
            extensions=(".npy",),
        ),
        batch_size=conf.batch_size,
    ):
        with torch.no_grad():
            outputs = non_max_suppression(
                model(inputs), 80, conf.conf_thresh, conf.nms_thresh
            )
        for detections, path in zip(outputs, paths):
            frame_ts = float(Path(path).name.rstrip(".npy"))
            if detections is not None:
                persons = post_process_detection(
                    detections, classes, npy_loader(path), conf
                )

                # ideally the below function should only return one person instead of a List
                if len(persons) > 0:
                    persons = get_only_biggest_person(persons)

                for person in persons:
                    person["frame"] = frame_ts
                    result_df = result_df.append(person, ignore_index=True)
            else:
                result_df = result_df.append(
                    {"x1": -1, "y1": -1, "x3": -1, "y3": -1, "frame": frame_ts},
                    ignore_index=True,
                )

    result_df.to_feather(output_file_path)

    return True


if __name__ == "__main__":
    config = Config(
        input_data_path="../data/pre_processor_output",
        output_data_path="../data/person_box_detector_output",
        run_id="ad5d61b0-1923-43eb-86ef-43b6c6f58001",
        conf_thresh=0.8,
        nms_thresh=0.4,
        img_size=416,
        img_height=416,
        img_width=739,
        config_path="../person_box_detector/config/yolov3.cfg",
        weights_path="../person_box_detector/config/yolov3.weights",
        class_path="../person_box_detector/config/coco.names",
        batch_size=128,
    )

    detect_persons(conf=config)
