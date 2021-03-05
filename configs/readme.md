# Config

## What?

The main aim of this `Config` class is to bundle all the settings and parameters for a single [`Job`](../pipeline/readme.md#Job)
in one instance having immutable attributes. The _immutability_ is important w.r.t. scalability and parallel execution
of [`Pipeline`](../pipeline/readme.md#Pipeline)s.
