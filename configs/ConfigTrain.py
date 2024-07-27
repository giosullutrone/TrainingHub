from typing import Union
from transformers import TrainingArguments
from recipes import DatasetRecipe, ModelRecipe, ModelTemplateRecipe, PeftRecipe, QuantizationRecipe, TokenizerRecipe
from cookbooks import DATASET_COOKBOOK, MODEL_COOKBOOK, MODEL_TEMPLATE_COOKBOOK, TOKENIZER_COOKBOOK, QUANTIZATION_COOKBOOK, PEFT_COOKBOOK
from dataclasses import dataclass, field


@dataclass
class ConfigTrain:
    """
    Configuration class for training by CLI.
    """

    # --------------------------------------------------------------------------
    # Here we define all the parameters for our training procedure ("train_cli.py")
    # Each parameter is a field. For each of them we specify the type, default and some useful metadata.
    # 
    # Metadata:
    #   - `description` (str): Description of the field. Used both as documentation and also by the argument parser
    #                          as information to show when calling `--help` on `train_cli.py` 
    #   - `required` (bool): Whether it is required for running `train_cli.py`. 
    #                        Note: This is done instead of removing `None` from (Union[..., None]) so that we can use also
    #                              use a starting config file for common parameters (See utils.parsers.get_config_from_argparser for the code)
    #   - `recipe_keywords` (List[str]): List of optional fields' names to use to create the recipe in case it was specified as a string.
    #                           For example --model_recipe "MistralModelRecipe" will also use `model_load` and `model_config` fields for initialization.
    #   - `cookbook` (CookBook): Cookbook to use to get the `recipe` if it was specified by string. 
    #                            For example --model_recipe "MistralModelRecipe", the string "MistralModelRecipe" would be used to get the 
    #                            class from the given cookbook.
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Required parameters
    # Paths
    dataset_name: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Huggingface path or local path of the dataset.", 
            "required": True
        }
    )
    model_name: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Huggingface path or local path of the model.", 
            "required": True
        }
    )
    tokenizer_name: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Huggingface path or local path of the tokenizer.", 
            "required": True
        }
    )

    # Recipes
    dataset_recipe: Union[DatasetRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for loading and preprocessing the dataset. Consult recipes.DatasetRecipe for all the available one.", 
            "required": True,
            "recipe_keywords": ("dataset_load", "dataset_response_template"),
            "cookbook": DATASET_COOKBOOK
        }
    )
    model_recipe: Union[ModelRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for loading and configuring the model. Consult recipes.ModelRecipe for all the available one.", 
            "required": True,
            "recipe_keywords": ("model_load", "model_config"),
            "cookbook": MODEL_COOKBOOK
        }
    )
    model_template_recipe: Union[ModelTemplateRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for the prompt template of the model. Consult recipes.ModelTemplateRecipe for all the available one.", 
            "required": True,
            "recipe_keywords": ("model_response_template", "system_template"),
            "cookbook": MODEL_TEMPLATE_COOKBOOK
        }
    )
    tokenizer_recipe: Union[TokenizerRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for loading and configuring the tokenizer. Consult recipes.TokenizerRecipe for all the available one.", 
            "required": True,
            "recipe_keywords": ("tokenizer_load", "tokenizer_config"),
            "cookbook": TOKENIZER_COOKBOOK
        }
    )
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Not required parameters
    # Recipes
    quantization_recipe: Union[QuantizationRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the quantization of the model (if needed).",
            "recipe_keywords": ("quantization_config", ),
            "cookbook": QUANTIZATION_COOKBOOK
        }
    )
    peft_recipe: Union[PeftRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the peft adapter to apply to the model (if needed).",
            "recipe_keywords": ("peft_config", ),
            "cookbook": PEFT_COOKBOOK
        }
    )

    # Recipe kwargs
    dataset_load: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for dataset load."
        }
    )
    model_load: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for model load."
        }
    )
    model_config: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for model configuration."
        }
    )
    tokenizer_load: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for tokenizer load."
        }
    )
    tokenizer_config: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for tokenizer configuration."
        }
    )
    quantization_config: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for quantization configuration."
        }
    )
    peft_config: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for peft configuration."
        }
    )
    finetuner_arguments: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for finetuner configuration. For more information check out trl.SFTTrainer for available arguments."
        }
    )
    dataset_response_template: Union[str, None] = field(
        default=None, 
        metadata={
            "description": "Response template for the dataset used."
        }
    )
    model_response_template: Union[str, None] = field(
        default=None, 
        metadata={
            "description": "Response template for the model used."
        }
    )
    system_template: Union[str, None] = field(
        default=None, 
        metadata={
            "description": "System template for the model used."
        }
    )
    validation_split_size: Union[float, None] = field(
        default=None, 
        metadata={
            "description": "Fraction of training set to use for validation in case the dataset specified does not have one."
        }
    )
    dataset_size: Union[int, None] = field(
        default=None, 
        metadata={
            "description": "Number of samples to limit the training dataset to (the validation_split_size of this will be taken if specified). Default: None -> No limits"
        }
    )
    training_arguments: Union[TrainingArguments, dict, None] = field(
        default=None, 
        metadata={
            "description": "Training arguments to pass to the trainer. For more information check out transformers.TrainingArguments for available arguments."
        }
    )
    num_examples: int = field(
        default=0, 
        metadata={
            "description": "Number of examples to provide for each prompt."
        }
    )
    completion_only: bool = field(
        default=True, 
        metadata={
            "description": "Whether to train the model only on completion (i.e. text after response template) or the full prompt."
        }
    )
    dynamic_examples: bool = field(
        default=True, 
        metadata={
            "description": "Whether to use dynamic examples for few-shot prompting or fixed ones."
        }
    )
    system_tuning: bool = field(
        default=False, 
        metadata={
            "description": "Whether to use the basic implementation of PromptTuning or the modified one that places the learned tokens after the system template."
        }
    )
    prompt_tuning_most_common_init: bool = field(
        default=False, 
        metadata={
            "description": "Whether to use the most common words in the dataset as starting point for prompt tuning."
        }
    )
    num_proc: int = field(
        default=None, 
        metadata={
            "description": "How many processes to use for dataset mapping."
        }
    )
    verbose: bool = field(
        default=False, 
        metadata={
            "description": "Whether to set the logger to DEBUG mode."
        }
    )
    # --------------------------------------------------------------------------