# Splitter

# What?

This part of the pipeline is responsible to cut the video into sections depending on the time 
of lyrics and change in the positions of persons in the video.

## Data Contract

### Input format

These are files that can be read from `data/input` and `data/detected_persons/` by the code to a key-value pair kind of a data structures. There are multiple options but 
for now we are going with `*.json` files. In general this would be a nested data structure.

#### lyrics.json

The root key of this data structure should be - "start_time, end_time". 
The value would be another key-value pair containing details about the texts of the lyrics. For example
```json
{
  "00:01.23456, 00:02.45678" : {
    "text": ["i", "am", "singing", "!"], // could be a continuous string like "i am singing !"
    "font_type": "Tahoma"
  },
  
  "00:04.23456, 00:06.45678" : {
    "text": ["i", "am", "dancing", "!"],
    "font_type": "Calibri"
  },
  ...
}
```

#### some_random_uuid.json

Same as above but the root key would be - "timestamp" of the frame.
The value would be the another key-value pair containing the specifications of the boxes where the persons were detected.

```json
{
  "00:01.23456" : {
    "x1": 24,
    "y1": 43,
    "x3": 56,
    "y3": 80
  },
  
  "00:02.23456" : {
    "x1": 24,
    "y1": 43,
    "x3": 56,
    "y3": 80
  },
  ...
}
```

### Output format

The data structure would be same as input format.

#### some_random_uuid.json

The root key of this JSON file is the range of time-stamps. For example

```json
{
  "00:01.23456, 00:02.45678": {
    "text": [
      "i",
      "am",
      "singing",
      "!"
    ],
    "forbidden_zones": {
      "x1": 24,
      "y1": 43,
      "x3": 56,
      "y3": 80
    }
  },
  "00:04.23456, 00:05.45678": {
    "text": [
      "i",
      "am",
      "singing",
      "!"
    ],
    "forbidden_zones": {
      "x1": 24,
      "y1": 43,
      "x3": 56,
      "y3": 80
    }
  },
  ...
}
```

