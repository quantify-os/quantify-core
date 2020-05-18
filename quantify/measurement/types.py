
def check_attribute(obj, attr, t):
    if not hasattr(obj, attr):
        raise AttributeError("{} does not have '{}'".format(obj, attr))
    if t and not isinstance(getattr(obj, attr), t):
        raise AttributeError("{} has attribute {} which should be of type str but is {}".format(obj, attr, t))


class Settable(type):
    """
    Defines the Settable concept, which is considered complete if the given type satisfies the following:

    contains attributes
        - set(float)
        - name: str
        - unit: str
    """
    def __new__(mcs, obj):
        for attr, t in {'name': str, 'unit': str, 'set': None}.items():
            check_attribute(obj, attr, t)
        return obj


class Gettable(type):
    """
    Defines the Gettable concept, which is considered complete if the given type satisfies the following:

    contains attributes
        - get()
        - name: str
        - unit: str
    """
    def __new__(mcs, obj):
        for attr, t in {'name': str, 'unit': str, 'get': None}.items():
            check_attribute(obj, attr, t)
        return obj
