import keyword
import re
import string


class DottedDict(dict):
    """
    Override for the dict object to allow referencing of keys as attributes, i.e. dict.key
    """

    def __init__(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                self._parse_input_(arg)
            elif isinstance(arg, list):
                for k, v in arg:
                    self.__setitem__(k, v)
            elif hasattr(arg, "__iter__"):
                for k, v in list(arg):
                    self.__setitem__(k, v)

        if kwargs:
            self._parse_input_(kwargs)

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DottedDict, self).__delitem__(key)
        del self.__dict__[key]

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        # Do this to match python default behavior
        except KeyError:
            raise AttributeError(attr)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        """
        Wrap the returned dict in DottedDict() on output.
        """
        return "{0}({1})".format(type(self).__name__, super(DottedDict, self).__repr__())

    def __setattr__(self, key, value):
        # No need to run _is_valid_identifier since a syntax error is raised if invalid attr name
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        try:
            self._is_valid_identifier_(key)
        except ValueError:
            if not keyword.iskeyword(key):
                key = self._make_safe_(key)
            else:
                raise ValueError('Key "{0}" is a reserved keyword.'.format(key))
        super(DottedDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def _is_valid_identifier_(self, identifier):
        """
        Test the key name for valid identifier status as considered by the python lexer. Also
        check that the key name is not a python keyword.
        https://stackoverflow.com/questions/12700893/how-to-check-if-a-string-is-a-valid-python-identifier-including-keyword-check
        """
        if re.match("[a-zA-Z_][a-zA-Z0-9_]*$", str(identifier)):
            if not keyword.iskeyword(identifier):
                return True
        raise ValueError('Key "{0}" is not a valid identifier.'.format(identifier))

    def _make_safe_(self, key):
        """
        Replace the space characters on the key with _ to make valid attrs.
        """
        key = str(key)
        allowed = string.ascii_letters + string.digits + "_"
        # Replace spaces with _
        if " " in key:
            key = key.replace(" ", "_")
        # Find invalid characters for use of key as attr
        diff = set(key).difference(set(allowed))
        # Replace invalid characters with _
        if diff:
            for char in diff:
                key = key.replace(char, "_")
        # Add _ if key begins with int
        try:
            int(key[0])
        except ValueError:
            pass
        else:
            key = "_{0}".format(key)
        return key

    def _parse_input_(self, input_item):
        """
        Parse the input item if dict into the dotted_dict constructor.
        """
        for key, value in input_item.items():
            if isinstance(value, dict):
                value = DottedDict(**{str(k): v for k, v in value.items()})
            if isinstance(value, list):
                _list = []
                for item in value:
                    if isinstance(item, dict):
                        _list.append(DottedDict(item))
                    else:
                        _list.append(item)
                value = _list
            self.__setitem__(key, value)

    def copy(self):
        """
        Ensure copy object is DottedDict, not dict.
        """
        return type(self)(self)

    def to_dict(self):
        """
        Recursive conversion back to dict.
        """
        out = dict(self)
        for key, value in out.items():
            if value is self:
                out[key] = out
            elif hasattr(value, "to_dict"):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                _list = []
                for item in value:
                    if hasattr(item, "to_dict"):
                        _list.append(item.to_dict())
                    else:
                        _list.append(item)
                out[key] = _list
        return out


class PreserveKeysDottedDict(dict):
    """
    Overrides auto correction of key names to safe attr names.  Can result in errors when using
    attr name resolution.
    """

    def __init__(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                self._parse_input_(arg)
            elif isinstance(arg, list):
                for k, v in arg:
                    self.__setitem__(k, v)
            elif hasattr(arg, "__iter__"):
                for k, v in list(arg):
                    self.__setitem__(k, v)

        if kwargs:
            self._parse_input_(kwargs)

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(PreserveKeysDottedDict, self).__delitem__(key)
        del self.__dict__[key]

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        # Do this to match python default behavior
        except KeyError:
            raise AttributeError(attr)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        """
        Wrap the returned dict in DottedDict() on output.
        """
        return "{0}({1})".format(
            type(self).__name__, super(PreserveKeysDottedDict, self).__repr__()
        )

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(PreserveKeysDottedDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def _parse_input_(self, input_item):
        """
        Parse the input item if dict into the dotted_dict constructor.
        """
        for key, value in input_item.items():
            if isinstance(value, dict):
                value = PreserveKeysDottedDict(**{str(k): v for k, v in value.items()})
            if isinstance(value, list):
                _list = []
                for item in value:
                    if isinstance(item, dict):
                        _list.append(PreserveKeysDottedDict(item))
                    else:
                        _list.append(item)
                value = _list
            self.__setitem__(key, value)

    def copy(self):
        """
        Ensure copy object is DottedDict, not dict.
        """
        return type(self)(self)

    def to_dict(self):
        """
        Recursive conversion back to dict.
        """
        out = dict(self)
        for key, value in out.items():
            if value is self:
                out[key] = out
            elif hasattr(value, "to_dict"):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                _list = []
                for item in value:
                    if hasattr(item, "to_dict"):
                        _list.append(item.to_dict())
                    else:
                        _list.append(item)
                out[key] = _list
        return out
