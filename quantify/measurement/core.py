
def check_attribute(obj, attr):
    if not hasattr(obj, attr):
        raise AttributeError("{} does not have '{}'".format(obj, attr))


class Settable(type):
    """
    Defines the Settable concept, which is considered complete if the
    given type satisfies the following:

    contains attributes
        - set
        - name
        - unit
    """
    def __new__(mcs, obj):
        for attr in ['set', 'name', 'unit']:
            check_attribute(obj, attr)
        return obj


class Gettable(type):
    """
    Defines the Gettable concept, which is considered complete if the
    given type satisfies the following:

    contains attributes
        - get
        - name
        - unit
    """
    def __new__(mcs, obj):
        for attr in ['get', 'name', 'unit']:
            check_attribute(obj, attr)
        return obj
