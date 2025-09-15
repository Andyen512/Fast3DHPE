# ddhpose/modeling/backbones/__init__.py
from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

# 当前目录
package_dir = Path(__file__).resolve().parent
# 遍历模块
for _, module_name, _ in iter_modules([str(package_dir)]):
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isclass(attribute):
            globals()[attribute_name] = attribute