from __future__ import annotations

'''
Class that stringifies your functions for a coding agent 
'''
import re
import os
import ast
from typing import Callable, Union, Any, Tuple

import inspect
import pkgutil
import importlib

def write_to_file(module_name: str = 'agents.functions.my_functions',
                  file_save_path: str = '/Users/BrianHsu/Desktop/GitHub/agents/agents/functions/my_functions.txt'
                  ) -> str:

    imported_module = sys.modules[module_name]
    stringified_objects = inspect.getsource(imported_module)

    with open(file_save_path, 'w') as file:
        file.write(stringified_objects)




