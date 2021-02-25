# lyrics_overlay

## What?

This project aims to put lyrics for a particular frame of a video in an optimal location avoiding 
overlap with the main subjects of the video.

Assumptions:
1. In this case "subjects" are "persons".
2. Number of subjects = 1
3. Maximum time-length = 5 minutes
4. Maximum frames per second = 60
5. Maximum resolution = 1920x1080 (FHD)
5. No time-overlapping lyrics

## Architecture

![flow](./docs/flow.svg)