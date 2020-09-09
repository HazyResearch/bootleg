'''
Taken from https://stackoverflow.com/questions/22046369/enforcing-class-variables-in-a-subclass/22047600
A metaclass that forces subclasses to instantiate certain attributes.
'''

def RequiredAttributes(*required_attrs):

    class RequiredAttributesMeta(type):
        def __init__(cls, name, bases, attrs):
            missing_attrs = ["'%s'" % attr for attr in required_attrs
                             if not hasattr(cls, attr)]
            if missing_attrs:
                raise AttributeError("class '%s' requires attribute%s %s" %
                                     (name, "s" * (len(missing_attrs) > 1),
                                      ", ".join(missing_attrs)))
    return RequiredAttributesMeta