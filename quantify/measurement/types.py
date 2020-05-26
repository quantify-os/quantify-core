def check_attribute(obj, attr, t):
    if not hasattr(obj, attr):
        raise AttributeError("{} does not have '{}'".format(obj, attr))
    if t and not isinstance(getattr(obj, attr), t):
        raise AttributeError("{} has attribute {} which should be of type str but is {}".format(obj, attr, t))


def check_control(obj):
    if hasattr(obj, "internal"):
        return obj.internal
    return True


class Settable:
    """
    Defines the Settable concept, which is considered complete if the given type satisfies the following:

    contains attributes
        - set(float)
        - name: str
        - label: str
        - unit: str

    optional attributes
        - internal (str): whether this parameter is internally or externally driven
    """
    def __init__(self, obj):
        for attr, t in {'name': str, 'unit': str, 'label': str, 'set': None}.items():
            check_attribute(obj, attr, t)
        self.__dict__.update(obj.__dict__)
        self.internal = check_control(obj)

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
        - label: str
        - unit: str

    optional attributes
        - internal (str): whether this parameter is internally or externally driven
    """

    def __init__(self, obj):
        for attr, t in {'name': str, 'unit': str, 'label': str, 'get': None}.items():
            check_attribute(obj, attr, t)
        self.__dict__.update(obj.__dict__)
        self.internal = check_control(obj)

    def prepare(self):
        pass

    def finish(self):
        pass
