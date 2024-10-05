from __future__ import annotations

'''
Class that stringifies your functions
'''
import re
from typing import Callable, Union, Any
import inspect
import ast
import pkgutil
import importlib

def extract_function_name(function: Callable) -> Union[None, str]:
    '''Gets the name of the function as a str'''
    pattern = r"def\s+(\w+)\s*\("
    match = re.search(pattern, function)
    if match:
        return match.group(1)  # Return the captured function name
    return None

def func_to_string(functions: Callable | list[Callable]) -> dict[str, str]:
    '''Converts a callable function into a str
    Inputs: 
    ======
    function: 
        sequence of callables / functions to convert into str 

    Outputs: 
    =======
    function_lists: 
        a list of functions as str
    '''
    if not isinstance(functions, list):
        functions = [functions]

    func_infos: dict[str, str] = [{
        'name': inspect.getsource(function),
        'full_description': extract_function_name(function)
    } for function in functions]

    return func_infos

def get_functions_from_package(package):
    functions = []
    # Ensure the package has a __path__ attribute
    if hasattr(package, '__path__'):
        for module_info in pkgutil.iter_modules(package.__path__):
            # Import the module
            module = module_info.module_finder.find_module(module_info.name).load_module(module_info.name)
            # Collect functions defined in the module
            functions.extend([name for name, obj in inspect.getmembers(module, inspect.isfunction)])
    else:
        print(f"{package.__name__} does not have a __path__ attribute. It may not be a package.")
    return functions

def get_all_submodules(package_name):
    """
    Recursively get all submodules from a package.
    
    Args:
        package_name (str): The name of the package to search.
        
    Returns:
        List of submodule names.
    """
    submodules = []
    # Dynamically import the package using its name
    package = importlib.import_module(package_name)
    # Helper function to iterate through submodules
    def recurse_modules(package):
        # Iterate through all modules in the package
        for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            submodules.append(module_info.name)  # Append the module name
            if module_info.ispkg:  # Check if it's a package
                # If it's a package, recurse into it
                recurse_modules(importlib.import_module(module_info.name))
    # Start recursion with the main package
    recurse_modules(package)
    return submodules


def get_functions_from_submodules(start_module_name):
    """
    Recursively get all functions from submodules starting from a given module.
    
    Args:
        start_module_name (str): The name of the starting submodule.
        
    Returns:
        Dictionary of submodule names with their respective functions.
    """
    functions_dict = {}

    # Dynamically import the starting module
    start_module = importlib.import_module(start_module_name)

    # Helper function to recursively find functions in submodules
    def recurse_modules(module):
        module_name = module.__name__
        functions = [name for name, obj in inspect.getmembers(module, inspect.isfunction)]
        
        if functions:
            functions_dict[module_name] = functions  # Store functions in the dictionary

        # Iterate through submodules
        for module_info in pkgutil.iter_modules(module.__path__, module_name + '.'):
            submodule = importlib.import_module(module_info.name)
            recurse_modules(submodule)  # Recursively call for the submodule

    # Start recursion with the initial module
    recurse_modules(start_module)

    return functions_dict

class FunctionReader:
    '''Gets all of the functions from a package, then turns it into a string'''

    def __init__(self, function_registry: Any, class_registry: Any, sub_module: Any) -> None:
        pass
    
    def stringify_file(self, file_path: str) -> list[dict[str, str]] | dict[str, str]: 
        '''Turns all of the functions in a file into a list of strings
        
        Inputs: 
        ======
        file_path: str 
            .py file path that we want to turn into a string 
        
        Outputs: 
        =======
        func_infos: list[str]
            list of each function's raw 
        '''
        try: 
            with open(file_path, 'r') as file:
                file_content = file.read()
            tree = ast.parse(file_content)
            functions:list[Callable] = [node.name for node in ast.walk(tree) 
                                        if isinstance(node, ast.FunctionDef)]
            func_infos: list[dict[str, str]] = func_to_string(functions)
            return func_infos
            
        except Exception as e: 
            return {'error': f'Error extracting functions from file: {e}'}
    
    def stringify_registry(self, package_name: ...) -> dict[str, str]: 
        imported_module: str
        pkgutil.iter_modules(imported_module.__path__)
        
        return ...

    def pull_from_registry(self, function_register):
        '''

        '''
        ...

if __name__ == "__main__":
    
    from agents import generator_registry
    from agents.generators import *
    
    registry = generator_registry

    func_str = inspect.getsource(FunctionReader)
    breakpoint()
