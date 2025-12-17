from typing import Callable, Dict, Optional

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._map: Dict[str, Callable] = {}

    def register(self, name: Optional[str] = None):
        def deco(obj):
            key = name or getattr(obj, "__name__", None)
            if key is None:
                raise ValueError("Cannot register unnamed object.")
            if key in self._map:
                raise KeyError(f"{self.name} already has key: {key}")
            self._map[key] = obj
            return obj
        return deco

    def get(self, key: str):
        if key not in self._map:
            raise KeyError(f"{self.name} missing key: {key}. Registered: {list(self._map.keys())}")
        return self._map[key]

MODELS = Registry("MODELS")
HEADS  = Registry("HEADS")
LOSSES = Registry("LOSSES")
