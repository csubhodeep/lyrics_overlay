# Pipeline

## What?

This part of the repo defines the basic components which enables us to realize the implementation of the architecture.

Here we define two main abstractions -
1. Job
2. Pipeline

### Job

The `Job` is basically an abstraction of a function that could be executed any time either in a stand-alone way or
in a flow together with other `Job`s. Every instance of `Job` must be constructed using a `Config` object and
a function that takes the `Config` object as the only input.

This is very important and thereby gives this flexibility to add as many `Job`s in a `Pipeline` and execute them in a
certain order.

### Pipeline

The `Pipeline` is an abstraction of a functionality/procedure/process that 'connects' a bunch of `Job`s together and
allows executing them one after the another.

## Why?

This was required in order to make sure that all steps of a pipeline run smoothly in coordination with other steps.

## How?

Each instance of `Pipeline` requires a unique id and sets the same to all the `Config` objects of each `Job` in it.
