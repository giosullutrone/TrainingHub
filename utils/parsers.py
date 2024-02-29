import argparse
from dataclasses import fields
from typing import Union, Dict, Iterable, List, Set
import importlib.util
from ast import literal_eval
from configs import Config


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
    is_set = set()

    class IsStored(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            is_set.add(self.dest) # save to global reference
            setattr(namespace, self.dest, values) # implementation of store_action

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
        parser.add_argument(f"--{field.name}", type=field_type, default=field_default, help=field_help, action=IsStored)

    return parser, is_set


def get_recipe_kwargs(field_name: str, recipe_keywords: Union[Dict, Iterable], config: Config, kwargs: Dict, is_set: Set):
    # If the keywords are not given as dict but as Iterable,
    # => Convert them to a dict.
    if not isinstance(recipe_keywords, dict): recipe_keywords = {x: x for x in recipe_keywords}
    # Generate the recipe kwargs
    recipe_kwargs = {}
    for keyword in recipe_keywords:
        config_field = recipe_keywords[keyword]
        # For each keyword, get the corresponding dict or create an empty one from the config
        # And merge the one provided by CLI with the config one.

        # If the field has been set by CLI then it has priority compared to config provided, else the opposite
        recipe_args_from_config = getattr(config, config_field) if getattr(config, config_field, None) is not None else {}
        recipe_args_from_cli = kwargs[config_field] if kwargs[config_field] is not None else {}

        if field_name in is_set: recipe_kwargs[keyword] = {**recipe_args_from_config, **recipe_args_from_cli}
        else: recipe_kwargs[keyword] = {**recipe_args_from_cli, **recipe_args_from_config}

        if not recipe_kwargs[keyword]: recipe_kwargs[keyword] = None
    return recipe_kwargs


def get_field_args(field_name: str, config: Config, kwargs: Dict, is_set: Set):
    if getattr(config, field_name, None) is not None:
        if isinstance(getattr(config, field_name), dict):
            if field_name in is_set: arg = {**getattr(config, field_name, {}), **kwargs[field_name]}
            else: arg = {**kwargs[field_name], **getattr(config, field_name, {})}
        elif field_name in is_set: arg = kwargs[field_name]
        else: arg = getattr(config, field_name)
    else: arg = kwargs[field_name]
    return arg


def get_config_from_argparser(cls: Config) -> Config:
    """
    Parse command-line arguments and generate a Config instance.

    Returns:
        Config: Configuration instance based on command-line arguments.
    """
    # Generate an argument parser for the Config dataclass
    parser, is_set = generate_argparser_from_dataclass(cls, description="Train the model with the given configuration, be it from terminal or config file.")
    # Add a specific argument for the configuration file path
    parser.add_argument("-c", "--config_file", required=False, default=None, type=str, help="Path to the Python file containing base configuration. Example can be found in the configs folder.")
    # Parse command-line arguments
    args = parser.parse_args()
    config_file_path = args.config_file
    kwargs = vars(args)

    # Load configuration from the specified file
    if config_file_path: config = load_config(config_file_path)
    else: config = cls()

    # For each field join config and CLI fields
    for field in fields(cls):
        # If the field has been specified
        if kwargs[field.name]:
            # If the field is a recipe, we need to instantiate the recipe
            if "recipe" in field.metadata:
                # Extract recipe information from metadata
                recipe_name = kwargs[field.name]
                recipe_keywords = field.metadata["recipe"]
                cookbook = field.metadata["cookbook"]
                setattr(config, field.name, cookbook.get(recipe_name)(**get_recipe_kwargs(field.name, recipe_keywords, config, kwargs, is_set)))
            else:
                # Set other fields
                arg = get_field_args(field.name, config, kwargs, is_set)
                if "cls" in field.metadata: setattr(config, field.name, field.metadata["cls"](**arg))
                else: setattr(config, field.name, arg)
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
    for field in fields(config):
        # Check for required fields
        if field.metadata.get("required", False) and getattr(config, field.name) is None:
            raise ValueError(f"{field.name} is required but not provided from config file or terminal.")

        # Check if a recipe field is of type string
        if "recipe" in field.metadata and isinstance(getattr(config, field.name), str):
            raise ValueError(f"{field.name} is of type string even though it is marked as recipe.")
