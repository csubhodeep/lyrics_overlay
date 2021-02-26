# pre-processor

## What?

This step of the pipeline is responsible to process the input
video and lyrics into formats compatible and efficient
with the rest of the pipeline.

## Data Contract

The `data_fetcher.py` code only requires link to the video and lyrics files from where it can fetch them.

Below we are specify the data contract for the `sampler.py` only.

### Input

#### some_random_uuid.csv

This is the file containing the details of the lyrics having the below format
```
start_time,end_time,text,font_type
00:03.1,00:05.5,"main gaa raha hoon !",""
```

#### some_random_uuid.mp4

This is the video file containing the total video.

### Output

#### some_random_uuid.feather

This file contains the details of the lyrics having the below format but rows are sorted by `start_time`
```
start_time,end_time,text,font_type
3100,5500,"main gaa raha hoon !",""
```

#### some_random_uuid/*.npy

These files are the frames resized, sampled and stored with their name as timestamps (in ms) of the frames.
