# Optimizer

This step is responsible to find out the best location of the 
lyrics box for each frame.

## Data Contract

### Input format

This is a file containing the forbidden zones in the frame
and the corresponding text to be placed in that frame.

#### some_random_uuid.feather

```
start_time,end_time,text,font_type,x1,y1,x3,y3
1100,1300,"i love you !","",24,42,56,76
```

### Output format

This is a file that contains the optimum location (and thereby the size)
of the lyrics box and the font-size for each frame in the input dataset

#### some_random_uuid.feather

```
start_time,end_time,text,font_type,x1,y1,x3,y3,x1_opti,y1_opti,x3_opti,y3_opti,font_size
1100,1300,"i love you !","",24,42,56,76,24,42,56,76,2.5
```