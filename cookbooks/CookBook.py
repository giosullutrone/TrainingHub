# Modified from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/registry.py


class CookBook:
    """
    The cookbook that provides name -> object mapping, to support third-party users' custom modules.

    To create a cookbook (e.g. a backbone cookbook):

    .. code-block:: python

        BACKBONE_COOKBOOK = CookBook('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_COOKBOOK.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_COOKBOOK.register(MyBackbone)
    """

    def __init__(self, name: str):
        self.name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (name not in self._obj_map), (f"An object named '{name}' was already registered in {self.name} cookbook")
        self._obj_map[name] = obj
        setattr(self, f"get_{name}", lambda: obj)

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None: raise KeyError(f"No object named '{name}' found in {self.name} cookbook!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()
