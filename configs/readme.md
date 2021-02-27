# Config

## What?

This is a class that inherits from [`collections.UserDict`](https://docs.python.org/3/library/collections.html#collections.UserDict)
class of Python std-lib.

## Why?

The main aim of this `Config` class is to bundle all the settings and parameters for a single [`Job`](../pipeline/readme.md#Job)
in one instance having immutable attributes. The _immutability_ is important w.r.t. scalability and parallel execution
of [`Pipeline`](../pipeline/readme.md#Pipeline)s.

## Notes about immutability

In Python the feature of immutability does not come inherent with the standard distribution of the language unlike some
other languages like Java, C++ etc. The following are some of the main features of immutability:
1. No value of attributes should be modifiable once set.
2. No new attributes should be added on the fly once the instance of the class is initialized.
3. No private/protected attributes should be accessible outside the instance/class.
