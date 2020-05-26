from enum import Enum


def check_attribute(obj, attr, t):
    if not hasattr(obj, attr):
        raise AttributeError("{} does not have '{}'".format(obj, attr))
    if t and not isinstance(getattr(obj, attr), t):
        raise AttributeError("{} has attribute {} which should be of type str but is {}".format(obj, attr, t))


class Datasource(Enum):
    INTERNAL = 1
    EXTERNAL = 2


class Settable:
    """
    Defines the Settable concept, which is considered complete if the given type satisfies the following:

    contains attributes
        - set(float)
        - name: str
        - unit: str
    """
    def __init__(self, obj, datasource):
        for attr, t in {'name': str, 'unit': str, 'set': None}.items():
            check_attribute(obj, attr, t)
        self.__dict__.update(obj.__dict__)
        self.datasource = datasource

    def prepare(self, setpoints):
        pass

    def finish(self):
        pass


class Gettable:
    """
    Defines the Gettable concept, which is considered complete if the given type satisfies the following:

    contains attributes
        - get()
        - name: str
        - unit: str
    """

    def __init__(self, obj, datasource):
        for attr, t in {'name': str, 'unit': str, 'get': None}.items():
            check_attribute(obj, attr, t)
        self.__dict__.update(obj.__dict__)
        self.datasource = datasource

    def prepare(self):
        pass

    def finish(self):
        pass
