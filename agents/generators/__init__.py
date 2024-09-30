from __future__ import annotations

'''Import registrated generator classes'''

from agents import generator_registry
from agents.registry.registry import import_submodules

from agents.generators.argo_generator import *

import_submodules(__name__) # import all files from generators


breakpoint()
