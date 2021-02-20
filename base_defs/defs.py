"""
This file contains the definitions of classes that are used through out the project
"""

class BaseImmutable:
    """Since in Python it is very difficult to ensure immutability of classes therefore any class that requires
    immutability as a feature should inherit from this class. By the term 'immutability' we wish to achieve
    the following things:
        1. No attributes should be added on the fly after the objects of the class has been constructed/initialised/instantiated
        2. The value of the attributes of the class once set should not be modifiable.
        3. No private attribute should be accessible from outside.
    """
    def __init__(self):
        self.status = None
