
def check_attribute(obj, attr):
    if not hasattr(obj, attr):
        raise AttributeError("{} does not have {}".format(obj, attr))


class Settable(type):
    def __new__(cls, obj):
        for attr in ['set', 'name', 'unit']:
            check_attribute(obj, attr)
        return obj


class Gettable(object):
    def __new__(cls, obj):
        for attr in ['get', 'name', 'unit']:
            check_attribute(obj, attr)
        return obj
