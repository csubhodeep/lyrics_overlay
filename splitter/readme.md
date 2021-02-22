# Splitter

## What?

This part of the pipeline is responsible to cut the video into sections depending on the time 
of lyrics and change in the positions of persons in the video.

## Data Contract

### Input format

#### some_random_uuid.feather

This is a file should be read from `data/pre_processed_output`. It would be containing the details of the lyrics sorted w.r.t `start_time`.

```
start_time,end_time,text,font_type
00:03.1,01:20.5,"main gaa raha hoon !",""
```

#### some_random_uuid.feather

This file should be read from `data/detected_persons_output`.
The values in this file would be containing the specifications of the boxes where the persons were detected in the frames with the corresponding timestamp.

```
timestamp,x1,x3,y1,y3
00:01.23456,24,43,56,80 
```

### Output format

The data structure would be same as input format.

#### some_random_uuid.feather

```
start_time,end_time,text,font_type,x1,x3,y1,y3
00:01.23456,00:02.45678,"i am singing !","",24,43,56,80
```
