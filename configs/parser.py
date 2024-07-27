from dataclasses import fields
from typing import Union, Dict
from configs import ConfigTrain
from configs.parser_utils import load_config, generate_argparser_from_dataclass
from transformers import TrainingArguments


def get_config_from_argparser(cls: ConfigTrain) -> ConfigTrain:
    """
    Parse command-line arguments and generate a Config instance.

    Returns:
        Config: Configuration instance based on command-line arguments.
    """
    # Generate an argument parser for the Config dataclass
    parser = generate_argparser_from_dataclass(cls, description="Train the model with the given configuration, be it from terminal or config file.")
    # Add a specific argument for the configuration file path
    parser.add_argument("-c", "--config_file", required=False, default=None, type=str, help="Path to the Python file containing base configuration. Example can be found in the configs folder.")

    # Parse command-line arguments
    cmd = parser.parse_args()
    config_file_path = cmd.config_file

    if config_file_path:
        # Load configuration from the specified file
        config: ConfigTrain = load_config(config_file_path)
        # Get kwargs from cmd
        cmd_kwargs = vars(cmd)

        for field in fields(config):
            if "recipe" in field.name:
                config = set_recipe(config, field.name, cmd_kwargs)
                
            elif "training_arguments" == field.name:
                config = set_class(TrainingArguments, config, field.name, cmd_kwargs)
                
            else:
                # Set other fields
                if isinstance(cmd_kwargs[field.name], dict):
                    setattr(config, field.name, join_dicts(getattr(cmd_kwargs[field.name], config, field.name)))
                else: 
                    if cmd_kwargs[field.name]: setattr(config, field.name, cmd_kwargs[field.name])
    else:
        # If no configuration file is provided, create a Config instance from command-line arguments
        kwargs = vars(cmd_kwargs); kwargs.pop("config_file")
        if cls == ConfigTrain:
            config = ConfigTrain(training_arguments=TrainingArguments(**kwargs.pop("training_arguments")), **kwargs)

    # Validate the configuration
    validate_config(config)
    return config

def join_dicts(priority: Union[Dict, None], secondary: Union[Dict, None]) -> Dict:
    if not secondary: secondary = {}
    if not priority: priority = {}
    return {**secondary, **priority}

def set_recipe(config: ConfigTrain, field: str, cmd_kwargs: Dict) -> ConfigTrain:
    # Extract recipe information from metadata
    recipe_keywords = field.metadata["recipe_keywords"]
    cookbook = field.metadata["cookbook"]
    
    # Extract information from config and cmd
    recipe_cmd_name = cmd_kwargs[field]
    recipe_config_name = getattr(config, field)
    
    # Extract the kwargs from config and cmd
    recipe_cmd_kwargs = {x: cmd_kwargs[x] for x in recipe_keywords}
    recipe_config_kwargs = {x: getattr(config, x) for x in recipe_keywords}
    
    # Make sure that the recipe contains a non-empty string
    assert isinstance(recipe_config_name, str) and recipe_config_name
    
    # If the names do not correspond, use the name and kwargs from cmd
    if recipe_cmd_name and recipe_cmd_name != recipe_config_name:
        setattr(config, field, cookbook.get(recipe_cmd_name)(**recipe_cmd_kwargs)); return
        
    # For each recipe's keyword, join the dict from cmd and config while giving priority to cmd
    recipe_kwargs = {recipe_keyword: join_dicts(recipe_cmd_kwargs[recipe_keyword], recipe_config_kwargs[recipe_keyword]) for recipe_keyword in recipe_keywords}
    setattr(config, field, cookbook.get(recipe_config_name)(**recipe_kwargs)); return config

def set_class(cls, config: ConfigTrain, field: str, cmd_kwargs: Dict) -> ConfigTrain:        
    # Extract the kwargs from config and cmd
    cls_cmd_kwargs = cmd_kwargs[field]
    cls_config_kwargs = getattr(config, field)
    
    # Init the cls by joining the kwargs dict from cmd and config while giving priority to cmd
    setattr(config, field, cls(**join_dicts(cls_cmd_kwargs, cls_config_kwargs))); return config

def validate_config(config: ConfigTrain):
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
