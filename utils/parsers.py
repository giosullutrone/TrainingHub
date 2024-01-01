from configs import Config
import argparse
from dataclasses import fields
import importlib.util
from cookbooks import CookBook
from transformers import TrainingArguments
from typing import Union, Dict
import ast


def load_config(config_path: str) -> Config:
    # Load the configuration file dynamically
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Get the Config instance from the loaded module
    config = getattr(config_module, "config", None)

    if config is None:
        raise ValueError("Config instance not found in the specified module")

    if not isinstance(config, Config):
        raise ValueError(f"The loaded instance is not of type {Config}")
    return config


def generate_argparser_from_dataclass(dataclass_type, description: str):
    parser = argparse.ArgumentParser(description=description)

    for field in fields(dataclass_type):
        field_type = field.type
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            valid_types = [t for t in field_type.__args__ if t in (str, int, dict, Dict, float)]
            field_type = valid_types[0]
            if field_type == dict or field_type == Dict: field_type = ast.literal_eval
        field_default = field.default if field.default is not field.default_factory else None
        field_help = field.metadata.get("description", "")
        parser.add_argument(f"--{field.name}", type=field_type, default=field_default, help=field_help)
    return parser


def get_config_from_argparser() -> Config:
    # --------------------------------------------------------------------------
    # ### Argparse
    # Here we capture the config file and any argument needed from the CLI
    # Create the parser
    parser = generate_argparser_from_dataclass(Config, description="Train the model with configurations from a specified Python configuration file.")
    # Add the config file
    parser.add_argument(
        "-c", "--config_file",
        required=False, default=None, type=str,
        help="Path to the Python file containing configuration for training. Parameters can be over overridden by passing them by terminal. Example can be found in the configs folder."
    )

    # Capture the arguments
    args = parser.parse_args()
    config_file_path = args.config_file
    print(config_file_path)

    # Load the configuration instance if provided
    if config_file_path is not None:
        config: Config = load_config(config_file_path)
        args_dict: dict = vars(args)
        # Override the config's variables if given
        for field in fields(Config):
            if args_dict[field.name] is None: continue
            if "recipe" in field.metadata:
                keywords = field.metadata["recipe"]
                cookbook: CookBook = field.metadata["cookbook"]
                recipe_name = args_dict[field.name]
                setattr(config, field.name, cookbook.get(recipe_name)(**{x: args_dict[x] for x in keywords}))
            elif field.name == "training_arguments":
                config.training_arguments = TrainingArguments(**args_dict["training_arguments"])
            else: setattr(config, field.name, args_dict[field.name])
    # Else use the args from the parser to generate one
    else:
        kwargs = vars(args)
        kwargs.pop("config_file")
        config: Config = Config(training_arguments=TrainingArguments(**kwargs.pop("training_arguments")), **kwargs)

    # Verify that all the required variables have been given (whether from a config file or from terminal)
    for field in fields(Config):
        if field.metadata.get("required", False):
            assert getattr(config, field.name) is not None, f"{field.name} has not been provided neither from config file nor terminal but it is marked as required."
        if "recipe" in field.metadata:
            assert not isinstance(getattr(config, field.name), str), f"{field.name} is of type string even though it is marked as recipe."
    return config
    # --------------------------------------------------------------------------
