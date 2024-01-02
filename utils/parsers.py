import argparse
from dataclasses import fields
from typing import Union, Dict
import importlib.util
from ast import literal_eval
from configs import Config
from transformers import TrainingArguments


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
        raise ValueError("Invalid configuration module or Config instance not found")

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


def get_config_from_argparser() -> Config:
    """
    Parse command-line arguments and generate a Config instance.

    Returns:
        Config: Configuration instance based on command-line arguments.
    """
    # Generate an argument parser for the Config dataclass
    parser = generate_argparser_from_dataclass(Config, description="Train the model with configurations from a specified Python configuration file.")
    # Add a specific argument for the configuration file path
    parser.add_argument("-c", "--config_file", required=False, default=None, type=str, help="Path to the Python file containing configuration for training. Example can be found in the configs folder.")

    # Parse command-line arguments
    args = parser.parse_args()
    config_file_path = args.config_file

    if config_file_path:
        # Load configuration from the specified file
        config = load_config(config_file_path)
        args_dict = vars(args)

        for field in fields(Config):
            if args_dict[field.name] is not None:
                if "recipe" in field.metadata:
                    # Extract recipe information from metadata
                    keywords = field.metadata["recipe"]
                    cookbook = field.metadata["cookbook"]
                    recipe_name = args_dict[field.name]
                    # Set the field with the cookbook recipe
                    kwargs = {x: args_dict[x] for x in keywords}
                    for x in keywords:
                        if isinstance(getattr(config, x), dict): kwargs = {**getattr(config, x), **kwargs}
                    setattr(config, field.name, cookbook.get(recipe_name)(**kwargs))
                elif field.name == "training_arguments":
                    # Set training arguments
                    if isinstance(config.training_arguments, dict): 
                        config.training_arguments = TrainingArguments(**{**config.training_arguments, **args_dict["training_arguments"]})
                    else: config.training_arguments = TrainingArguments(**args_dict["training_arguments"])
                else:
                    # Set other fields
                    if isinstance(getattr(config, field.name), dict):
                        setattr(config, field.name, {**getattr(config, field.name), **args_dict[field.name]})
                    else: setattr(config, field.name, args_dict[field.name])
    else:
        # If no configuration file is provided, create a Config instance from command-line arguments
        kwargs = vars(args); kwargs.pop("config_file")
        config = Config(training_arguments=TrainingArguments(**kwargs.pop("training_arguments")), **kwargs)

    # Validate the configuration
    validate_config(config)
    return config


def validate_config(config: Config):
    """
    Validate the provided configuration.

    Args:
        config (Config): Configuration instance.

    Raises:
        ValueError: If a required field is not provided or if a recipe field is of type string.
    """
    for field in fields(Config):
        # Check for required fields
        if field.metadata.get("required", False) and getattr(config, field.name) is None:
            raise ValueError(f"{field.name} is required but not provided from config file or terminal.")

        # Check if a recipe field is of type string
        if "recipe" in field.metadata and isinstance(getattr(config, field.name), str):
            raise ValueError(f"{field.name} is of type string even though it is marked as recipe.")
