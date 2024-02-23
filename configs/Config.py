from dataclasses import dataclass, field


@dataclass
class Config:
    # --------------------------------------------------------------------------
    # Here we define all the parameters for our procedures
    # Each parameter is a field. For each of them we specify the type, default and some useful metadata.
    # 
    # Metadata:
    #   - `description` (str): Description of the field. Used both as documentation and also by the argument parser
    #                          as information to show when calling `--help` on the procedure.
    #   - `required` (bool): Whether it is required for running the procedure. 
    #                        Note: This is done instead of removing `None` from (Union[..., None]) so that we can use also
    #                              use a starting config file for common parameters (See utils.parsers.get_config_from_argparser for the code)
    #   - `recipe` (List[str]): List of optional fields' names to use to create the recipe in case it was specified as a string.
    #                           For example --model_recipe "MistralModelRecipe" will also use `model_load` and `model_config` fields for initialization.
    #   - `cookbook` (CookBook): Cookbook to use to get the `recipe` if it was specified by string. 
    #                            For example --model_recipe "MistralModelRecipe", the string "MistralModelRecipe" would be used to get the 
    #                            class from the given cookbook.
    #   - `cls` (class): Class to instantiate with the arguments provided.
    # --------------------------------------------------------------------------
    pass