from pathlib import Path
import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision.transforms import Compose, Resize, Pad, ToTensor

from .models import Darknet
from configs.make_config import Config
from person_box_detector.utils.utils import non_max_suppression, load_classes


# Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor


def detect_persons(conf: Config) -> bool:
    # doing random stuff
    some_output = {}
    file_name = f"{conf.run_id}.json"
    file_path = Path(conf.output_data_path).joinpath(file_name)
    with open(file_path, 'w') as f:
        json.dump(some_output, f)

    return True

def detect_image(img, img_size, model):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = Compose([Resize((imh, imw)),
         Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


if __name__ == "__main__":
    config_path='config/yolov3.cfg'
    weights_path='config/yolov3.weights'
    class_path='config/coco.names'
    img_size=416
    conf_thres=0.8
    nms_thres=0.4

    # Load model and weights
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)
    # model.cuda()
    model.eval()
    classes = load_classes(class_path)

    cap = cv2.VideoCapture('../data/girls_like_you_small.mp4')

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    detected_persons = {}
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg, img_size, model)

        img = np.array(pilimg)
        #print(detections)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        if detections is not None :
            #tracked_objects = mot_tracker.update(detections.cpu())

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            list_of_persons = []
            for x1, y1, x3, y3, conf, cls_conf, cls_pred in detections:
                cls = classes[int(cls_pred)]
                if cls == 'person':
                    box_h = int(((y3 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x3 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    x3 = x1 + box_w
                    y3 = y1 + box_w
                    list_of_persons.append({
                        'x1': x1,
                        'y1': y1,
                        'x3': x3,
                        'y3': y3
                    }
                    )
                    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (11, 111, 11), 4)
            detected_persons[f"frame_{i}"] = tuple(list_of_persons)
            i = i + 1
                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                #bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                #ax.add_patch(bbox)
                #plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                #         bbox={'color': color, 'pad': 0})
        #         cls = classes[int(cls_pred)]
        #         if cls == 'person':
        #             cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (11,111,11), 4)
        #             #cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), (11,11,111), -1)
        #             cv2.putText(frame, 'human', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                     (255, 255, 255), 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if i%10 == 0:
            break


    cap.release()
    cv2.destroyAllWindows()

    with open('../data/detected_persons/result.json', 'w') as f:
        json.dump(detected_persons, f)