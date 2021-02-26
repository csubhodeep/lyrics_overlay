# PyTorch Object Detection and Tracking
Object detection in images, and tracking across video frames

1. download the weights file using the script in config folder 'download_weights.sh'
2. install the things that are required to run the project. (torch, cv2, PIL, numba etc)
3. put a video in the data folder
4. run the inference.py after providing the path of ur video inside the code.
5. ignore the sort.py file, it is about object tracking to make it real time. we can detect for every frame we dont want real time.

Full story at:
https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98

References:
1. YOLOv3: https://pjreddie.com/darknet/yolo/
2. Erik Lindernoren's YOLO implementation: https://github.com/eriklindernoren/PyTorch-YOLOv3
3. YOLO paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf
4. SORT paper: https://arxiv.org/pdf/1602.00763.pdf
5. Alex Bewley's SORT implementation: https://github.com/abewley/sort
6. Installing Python 3.6 and Torch 1.0: https://medium.com/@chrisfotache/getting-started-with-fastai-v1-the-easy-way-using-python-3-6-apt-and-pip-772386952d03
