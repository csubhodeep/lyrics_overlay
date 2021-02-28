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

Each instance of `Pipeline` requires a unique run id and sets the same to all the `Config` objects of each `Job` in it.

## Notes about immutability in Python

In Python the feature of _immutability_ does not come inherent with the standard distribution of the language unlike some
other languages like Java, C++ etc. The following are some of the main features of immutability:
1. No value of attributes should be modifiable once set.
2. No new attributes should be added on the fly once the instance of the class is initialized.
3. No private/protected attributes should be accessible outside the instance/class.

Unfortunately none of them is implemented out-of-the-box so we could solve them in the following way:
1. We need to override the default `__setattr__` function behaviour of the class and check if an attribute already exists
- if yes then raise an error.
2. We need a specified set of names of attributes defined as a class attribute so that the same is applicable and accessible
to all the instances of the class.
3. We could use "__" as a prefix to the names of the member variables.

## Why we need immutability?

We need immutability because we don't want the configuration of each `Job` to get modified on the fly externally as it
can change the behaviour and may disrupt the pipeline causing outputs to land up in different locations and
cause the subsequent `Job`s to fail.
