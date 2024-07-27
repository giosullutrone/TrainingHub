import argparse
from dataclasses import fields
from typing import Union
import importlib.util
from ast import literal_eval
from configs import ConfigTrain


def load_config(config_path: str) -> ConfigTrain:
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
    if config is None or not isinstance(config, ConfigTrain):
        raise ValueError("Invalid configuration module or config instance not found")

    return config


def generate_argparser_from_dataclass(dataclass_type, description: str):
    """
    Generate an argparse.ArgumentParser based on the fields of a dataclass.

    Args:
        dataclass_type: Type of the dataclass.
        description (str): Description for the argument parser.

    Returns:
        argparse.ArgumentParser: Argument parser populated with dataclass fields.
    """
    parser = argparse.ArgumentParser(description=description)

    # Iterate through fields of the dataclass
    for field in fields(dataclass_type):
        field_type = field.type
        # Handle Union types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            valid_types = [t for t in field_type.__args__ if t in (str, int, dict, float)]
            field_type = valid_types[0]
            if field_type == dict:
                field_type = literal_eval
        field_default = field.default if field.default is not field.default_factory else None
        field_help = field.metadata.get("description", "")
        parser.add_argument(f"--{field.name}", type=field_type, default=field_default, help=field_help)

    return parser