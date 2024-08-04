import os
import json
import argparse
import importlib.util
from ast import literal_eval
from typing import List, Tuple, Any, Optional
from cookbooks.CookBook import CookBook
from dataclasses import make_dataclass, field, fields, asdict
from recipes import DatasetRecipe, ModelRecipe, TokenizerRecipe, PeftRecipe, QuantizationRecipe, TrainRecipe


def camel_to_snake(camel_string: str):
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

def generate_argparser_from_recipes(recipe_dataclasses: List[object], description: str):
    """
    Generate an argparse.ArgumentParser based on the fields of a dataclass.

    Args:
        dataclass_type: Type of the dataclass.
        description (str): Description for the argument parser.

    Returns:
        argparse.ArgumentParser: Argument parser populated with dataclass fields.
    """
    parser = argparse.ArgumentParser(description=description)

    for recipe_dataclass in recipe_dataclasses:
        parser.add_argument(f"--{camel_to_snake(recipe_dataclass.__name__)}", 
                            type=str, default=None, 
                            help="Recipe's name from cookbook")
        
        # Iterate through fields of the dataclass
        for field in fields(recipe_dataclass):
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

def load_config(config_path: str, recipe_dataclasses: List[Tuple[object, bool, CookBook]]):
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
    if config is None or not isinstance(config, dict):
        raise ValueError("Invalid configuration or config instance not found")

    config = get_config(recipe_dataclasses)(**config)
    return config

def get_config(recipe_dataclasses: List[Tuple[object, bool, CookBook]]):
    fis = []
    for recipe, required, cookbook in recipe_dataclasses:
        fis += [(x.name, x) for x in fields(recipe)]
        fis += [(camel_to_snake(recipe.__name__), field(default=None, metadata = {"required": required, 
                                                                                  "description": "Recipe's name from cookbook",
                                                                                  "cookbook": cookbook, 
                                                                                  "recipe_keywords": fields(recipe)}))]
    return make_dataclass("Config", fields=fis)


import yaml
from dataclasses import fields, is_dataclass

def create_config_file(file_name, recipe_dataclasses: List[object]):
    """
    Creates a file with the given name and writes its contents to it in JSON format.

    Args:
        file_name (str): The name of the file.
        dataclass_instance (MyClass): An instance of MyClass that contains the data to be written.
    """

    # Get the path of the parent directory
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    # Construct the full path to the target folder
    target_folder_path = os.path.join(parent_dir, 'recipes', 'train')

    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
    
    # Get the full path of the file
    file_path = os.path.join(target_folder_path, file_name)
        
    dataclass_instance = get_config([(x, False, None) for x in recipe_dataclasses])
        
    if not is_dataclass(dataclass_instance):
        raise ValueError("Provided instance is not a dataclass")
    
    dataclass_dict = {}
    
    for field in fields(dataclass_instance):
        field_info = {
            "description": field.metadata.get("description", "No description provided"),
            "default": field.default if field.default != field.default_factory else "No default value"
        }
        dataclass_dict[field.name] = field_info

    with open(file_path, 'w') as yaml_file:
        yaml.dump(dataclass_dict, yaml_file, default_flow_style=False)


if __name__ == "__main__":
    create_config_file("config.yaml", 
                       recipe_dataclasses=[DatasetRecipe, 
                                           ModelRecipe, 
                                           TokenizerRecipe, 
                                           PeftRecipe, 
                                           QuantizationRecipe, 
                                           TrainRecipe])