from typing import Union
from recipes import DatasetRecipe, ModelRecipe, TokenizerRecipe, QuantizationRecipe, ModelTemplateRecipe, PeftRecipe
from cookbooks import DATASET_COOKBOOK, MODEL_COOKBOOK, MODEL_TEMPLATE_COOKBOOK, TOKENIZER_COOKBOOK, QUANTIZATION_COOKBOOK, PEFT_COOKBOOK
from dataclasses import field, dataclass
from .Config import Config
from transformers import GenerationConfig


@dataclass
class ConfigGenerateDataset(Config):
    """
    Configuration class for training by CLI.
    """

    # --------------------------------------------------------------------------
    # ### Generation parameters
    # Paths
    generation_dataset_name: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Huggingface path or local path of the dataset.", 
            "required": True
        }
    )
    generation_model_name: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Huggingface path or local path of the model.", 
            "required": True
        }
    )
    generation_tokenizer_name: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Huggingface path or local path of the tokenizer.", 
            "required": True
        }
    )
    generation_output_path: Union[str, None] = field(
        default=None,
        metadata={
            "required": True,
            "description": "Output path to save the generated dataset to."
        }
    )

    # Recipes
    generation_dataset_recipe: Union[DatasetRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for loading and preprocessing the dataset. Consult recipes.DatasetRecipe for all the available one.", 
            "required": True,
            "recipe": {"dataset_load": "generation_dataset_load", "dataset_response_template": "generation_dataset_response_template"},
            "cookbook": DATASET_COOKBOOK
        }
    )
    generation_model_recipe: Union[ModelRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for loading and configuring the model. Consult recipes.ModelRecipe for all the available one.", 
            "required": True,
            "recipe": {"model_load": "generation_model_load", "model_config": "generation_model_config"},
            "cookbook": MODEL_COOKBOOK
        }
    )
    generation_model_template_recipe: Union[ModelTemplateRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for the prompt template of the model. Consult recipes.ModelTemplateRecipe for all the available one.", 
            "required": True,
            "recipe": {"model_response_template": "generation_model_response_template", "model_system_template": "generation_model_system_template"},
            "cookbook": MODEL_TEMPLATE_COOKBOOK
        }
    )
    generation_tokenizer_recipe: Union[TokenizerRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for loading and configuring the tokenizer. Consult recipes.TokenizerRecipe for all the available one.", 
            "required": True,
            "recipe": {"tokenizer_load": "generation_tokenizer_load", "tokenizer_config": "generation_tokenizer_config"},
            "cookbook": TOKENIZER_COOKBOOK
        }
    )
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Not required parameters
    # Recipes
    generation_quantization_recipe: Union[QuantizationRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the quantization of the model (if needed).",
            "recipe": {"quantization_config": "generation_quantization_config"},
            "cookbook": QUANTIZATION_COOKBOOK
        }
    )
    generation_peft_recipe: Union[PeftRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the peft adapter to apply to the model (if needed).",
            "recipe": {"peft_config": "generation_peft_config"},
            "cookbook": PEFT_COOKBOOK
        }
    )
    generation_peft_path: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Path to the peft adapter to apply to the model (if needed).",
        }
    )

    # Recipe kwargs
    generation_dataset_load: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for dataset load."
        }
    )
    generation_model_load: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for model load."
        }
    )
    generation_model_config: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for model configuration."
        }
    )
    generation_tokenizer_load: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for tokenizer load."
        }
    )
    generation_tokenizer_config: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for tokenizer configuration."
        }
    )
    generation_quantization_config: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for quantization configuration."
        }
    )
    generation_peft_config: Union[dict, None] = field(
        default=None, 
        metadata={
            "description": "Kwargs for peft configuration."
        }
    )
    generation_dataset_response_template: Union[str, None] = field(
        default=None, 
        metadata={
            "description": "Response template for the dataset used."
        }
    )
    generation_model_response_template: Union[str, None] = field(
        default=None, 
        metadata={
            "description": "Response template for the model used."
        }
    )
    generation_model_system_template: Union[str, None] = field(
        default=None, 
        metadata={
            "description": "System template for the model used."
        }
    )
    generation_starting_size: Union[int, None] = field(
        default=None, 
        metadata={
            "description": "Number of samples to limit the training dataset to. Default: None -> No limits"
        }
    )
    generation_num_examples: int = field(
        default=8, 
        metadata={
            "description": "Number of static examples to provide for each prompt."
        }
    )
    num_proc: int = field(
        default=None, 
        metadata={
            "description": "How many processes to use for dataset mapping."
        }
    )
    generation_config: Union[GenerationConfig, dict, None] = field(
        default=None, 
        metadata={
            "description": "#TODO ",
            "cls": GenerationConfig
        }
    )
    verbose: bool = field(
        default=False, 
        metadata={
            "description": "Whether to set the logger to DEBUG mode."
        }
    )
    # --------------------------------------------------------------------------