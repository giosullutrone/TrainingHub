from dataclasses import fields, is_dataclass
from typing import Union, Dict, Optional, List, Tuple, Any
from configs.parser_utils import generate_argparser_from_config
from trl import SFTConfig
from recipes.Recipe import Recipe
from configs import Config
from configs.parser_utils import load_config, config_to_flat_dict


def get_config_from_argparser():
    """
    Parse command-line arguments and generate a Config instance.

    Returns:
        Config: Configuration instance based on command-line arguments.
    """
    # Generate an argument parser for the recipes
    parser = generate_argparser_from_config(description="Train the model with the given configuration, be it from terminal or config file.")
    # Add a specific argument for the configuration file path
    parser.add_argument("-c", "--config_file", required=False, default=None, type=str, help="Path to the Python file containing base configuration. Example can be found in the configs folder.")

    # Parse command-line arguments
    cmd = parser.parse_args()
    config_file_path = cmd.config_file

    if config_file_path:
        # Load configuration from the specified file
        config: Config = load_config(config_file_path)
        config_kwargs: dict = config_to_flat_dict(config)
        
        # Get kwargs from cmd
        cmd_kwargs = vars(cmd)

        for field in fields(config):
            assert is_dataclass(field.type)
            config = set_recipe(config, field, config_kwargs, cmd_kwargs)
    else:
        # If no configuration file is provided, create a Config instance from command-line arguments
        kwargs = vars(cmd_kwargs); kwargs.pop("config_file")
        config = Config(training_arguments=SFTConfig(**kwargs.pop("training_arguments")), **kwargs)

    return config

def join_values(priority: Optional[Any], secondary: Optional[Any]) -> dict:
    assert priority is None or secondary is None or (type(priority) == type(secondary))
    
    if ((priority is not None and isinstance(priority, dict)) or 
       (secondary is not None and isinstance(secondary, dict))):
        if secondary is None: secondary = {}
        if priority is None: priority = {}
        return {**secondary, **priority}
    
    if priority is not None: return priority
    return secondary

def set_recipe(config, field, config_kwargs: dict, cmd_kwargs: dict):
    # Extract recipe information from metadata
    recipe_keywords = [x.name for x in fields(field.type)]
    cookbook = field.metadata["cookbook"]
    
    # Extract information from config and cmd
    recipe_cmd_name = cmd_kwargs[field.name]
    recipe_config_name = config_kwargs[field.name]
    
    # Extract the kwargs from config and cmd
    recipe_cmd_kwargs = {x: cmd_kwargs[x] for x in recipe_keywords}
    recipe_config_kwargs = {x: config_kwargs[x] for x in recipe_keywords}
    
    # Make sure that the recipe contains a non-empty string
    assert isinstance(recipe_config_name, str) and recipe_config_name
    
    # If the names do not correspond, use the name and kwargs from cmd
    if recipe_cmd_name and recipe_cmd_name != recipe_config_name:
        setattr(config, field.name, cookbook.get(recipe_cmd_name)(**recipe_cmd_kwargs)); return
    
    # For each recipe's keyword, join the dict from cmd and config while giving priority to cmd    
    recipe_kwargs = {recipe_keyword: join_values(recipe_cmd_kwargs[recipe_keyword], recipe_config_kwargs[recipe_keyword]) for recipe_keyword in recipe_keywords}
    setattr(config, field.name, cookbook.get(recipe_config_name)(**recipe_kwargs)); return config

def set_class(cls, config, field: str, config_kwargs: dict, cmd_kwargs: dict):        
    # Extract the kwargs from config and cmd
    cls_cmd_kwargs = cmd_kwargs[field]
    cls_config_kwargs = config_kwargs[field]
    
    # Init the cls by joining the kwargs dict from cmd and config while giving priority to cmd
    setattr(config, field, cls(**join_values(cls_cmd_kwargs, cls_config_kwargs))); return config
