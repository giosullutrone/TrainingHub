import os
import json
import argparse
import importlib.util
from ast import literal_eval
from typing import List, Tuple, Any, Optional
from cookbooks.CookBook import CookBook
from dataclasses import make_dataclass, field, fields, asdict
import yaml
from dataclasses import fields, is_dataclass
import os
from typing import List
from recipes.Recipe import Recipe
from configs import Config


def camel_to_snake(camel_string: str) -> str:
    """
    Convert camelCase string to snake_case string.

    Args:
        camel_string (str): The input string in camelCase format.

    Returns:
        str: The output string in snake_case format.
    """
    result = ''
    for i, char in enumerate(camel_string):
        if char.isupper() and i > 0:
            result += '_' + char.lower()
        else:
            result += char.lower()
    return result

def generate_argparser_from_config(description: str):
    """
    Generate an argparse.ArgumentParser based on the fields of a dataclass.

    Args:
        dataclass_type: Type of the dataclass.
        description (str): Description for the argument parser.

    Returns:
        argparse.ArgumentParser: Argument parser populated with dataclass fields.
    """
    parser = argparse.ArgumentParser(description=description)

    for recipe_dataclass in [x for x in fields(Config)]:
        parser.add_argument(f"--{recipe_dataclass.name}", 
                            type=str, default=None, 
                            help="Recipe's name from cookbook.")
        
        # Iterate through fields of the dataclass
        for field in fields(recipe_dataclass.type):
            field_type = field.type
            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Optional:
                valid_types = [t for t in field_type.__args__ if t in (str, int, dict, float)]
                field_type = valid_types[0]
                # If type is dict, we make the argparse evaluate the string
                # This converts the string '{"example": "example value"}' into the final dict
                if field_type == dict: field_type = literal_eval
            field_default = field.default if field.default is not None else None
            field_help = field.metadata.get("description", "")
            parser.add_argument(f"--{field.name}", type=field_type, default=field_default, help=field_help)

    return parser

def load_config(config_path: str) -> Config:
    """
    Load configuration from a Python file.

    Args:
        config_path (str): Path to the Python file containing configuration.

    Returns:
        Config: Loaded configuration instance.
    """
    # Create a spec from the file location
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    # Create a module from the spec
    config_module = importlib.util.module_from_spec(spec)
    # Execute the module to populate it
    spec.loader.exec_module(config_module)

    # Get the 'config' attribute from the module
    config = getattr(config_module, "config", None)

    # Check if a valid Config instance is obtained
    if config is None or not isinstance(config, Config):
        raise ValueError("Invalid configuration or config instance not found")
    return config

def config_to_flat_dict(config: Config) -> dict:
    config_dict = {}
    for fi in fields(config):
        if is_dataclass(fi.type):
            config_dict[fi.name] = fi.type.__name__
            config_dict.update(config_to_flat_dict(getattr(config, fi.name)))
        else: 
            config_dict[fi.name] = getattr(config, fi.name)
    return config_dict
