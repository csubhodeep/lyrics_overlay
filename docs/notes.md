# Various modes of App

## High level modes

### Video
#### Center box: p2
#### Down box: p2
#### User defined box: p2
#### AI based box: p1

### Image
#### Center box: p3
#### Down box: p3
#### User defined box: p3

## Pipeline 1 : p1

AI and video. Current pipeline.

## pipeline 2: p2

Non AI and video. 
In this case we dont have to create npy files. we just need list of timestamps.
optimizer still have to give output in same format. t1 t2 and text and text box.
In this we have to bypass detector and splitter.
and optimize will also bypass but it will provide simple boxes based on 
non AI modes of videos.

## pipeline 3: p3

image based. in this preprocessor doesnt have to create npy again. 
by pass detection and splitter
optimizer will give text box based on non ai mode selected by user. the output will be 
in same format as now. it will use the lyrics df to create the output for each lyrics line.
p3 and p2 looks similar till now.
But now in the post processor u have just read the input frame (not video) and use the same frame again and again
to write the video. but what should be loop through ? audio = 5 minutes. 30 fps. 1 frame stays for
33 ms. so we loop from 0 to 5 minutes (900 frames) with a gap of 33ms. and for every iteration we do the same logic
what we have now. frame= input_image (instead of videocapture frame)

# IDEA: to find optimal box

Use simple conv kernal to find zones which are not noisy or which are have a lot of plan area 
without edges in them. we will get segmentation map. that map will have mines on some pixels and 
other pixels are fine. then we will have to find location and size of box that fits in this mask without hiting 
the mines. that box should be big enough. it should be in beautiful location (not on edges and corners)




