from typing import Union
from transformers import TrainingArguments
from recipes import DatasetRecipe, ModelRecipe, TokenizerRecipe, QuantizationRecipe, ModelTemplateRecipe, PeftRecipe
from cookbooks import DATASET_COOKBOOK, MODEL_COOKBOOK, MODEL_TEMPLATE_COOKBOOK, TOKENIZER_COOKBOOK, QUANTIZATION_COOKBOOK, PEFT_COOKBOOK
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Configuration class for training by CLI.

    Attributes:
        - `model_recipe` (ModelRecipe): Recipe for configuring the model.
        - `model_template_recipe` (ModelTemplateRecipe): Recipe for configuring the model's template.
        - `tokenizer_recipe` (TokenizerRecipe): Recipe for configuring the tokenizer.
        - `dataset_recipe` (Union[DatasetRecipe, None]): Recipe for loading and preprocessing the dataset. If not specified, a YAML based dataset will be automatically selected.
        - `yaml_path` (Union[str, None]): Path to a YAML file compatible with lm-evaluation-harness library (check out the original library's github page for format and config files). Used only if no dataset recipe has been provided. 
        - `response_template` (Union[str, None]): Response template to use for completion-only training. Used only if no dataset recipe has been provided. TODO: add readme page with explanation of completion-only.
        - `quantization_recipe` (Union[QuantizationRecipe, None]): Recipe for model quantization (if applicable).
        - `peft_recipe` (Union[PeftRecipe, None]): Recipe for performance enhancement and fine-tuning (if applicable).
        - `dataset_name` (Union[str, None]): Name or path of the dataset. If not specified here, can be also given by terminal.
        - `model_name` (Union[str, None]): Name or path of the model. If not specified here, can be also given by terminal.
        - `validation_split_size` (Union[float, None]): Size of the validation split as a fraction of the dataset size. Used only if validation set doesn't exist for the dataset.
        - `num_examples` (int): Number of examples to give each prompt. They will be taken from the first elements num_examples elements of the train set. Default: 0.
        - `training_arguments` (Union[TrainingArguments, None]): Arguments for training the model (if applicable).
        - `finetuner_arguments` (Union[dict, None]): Arguments for fine-tuning the model (if applicable).
        - `completion_only` (bool): Whether to train on prompt completion only or to predict the whole prompt. Default: True.
        - `system_tuning` (bool): Whether to train `prompt tuning` on system prompts instead of the naive prepending. Only applicable if using `prompt tuning` as peft Default: False.
    """
    # Required
    dataset_name: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the model.", 
            "required": True
        }
    )
    model_name: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the model.", 
            "required": True
        }
    )
    tokenizer_name: Union[str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the model.", 
            "required": True
        }
    )

    dataset_recipe: Union[DatasetRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the model.", 
            "required": True,
            "recipe": ("dataset_load", "dataset_response_template"),
            "cookbook": DATASET_COOKBOOK
        }
    )
    model_recipe: Union[ModelRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the model.", 
            "required": True,
            "recipe": ("model_load", "model_config"),
            "cookbook": MODEL_COOKBOOK
        }
    )
    model_template_recipe: Union[ModelTemplateRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the model.", 
            "required": True,
            "recipe": ("model_response_template", "system_template"),
            "cookbook": MODEL_TEMPLATE_COOKBOOK
        }
    )
    tokenizer_recipe: Union[TokenizerRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the model.", 
            "required": True,
            "recipe": ("tokenizer_load", "tokenizer_config"),
            "cookbook": TOKENIZER_COOKBOOK
        }
    )

    # Not required
    quantization_recipe: Union[QuantizationRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the model.",
            "recipe": ("quantization_config", ),
            "cookbook": QUANTIZATION_COOKBOOK
        }
    )
    peft_recipe: Union[PeftRecipe, str, None] = field(
        default=None,
        metadata={
            "description": "Recipe for configuring the model.",
            "recipe": ("peft_config", ),
            "cookbook": PEFT_COOKBOOK
        }
    )
    # Params not required
    dataset_load: Union[dict, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})
    model_load: Union[dict, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})
    model_config: Union[dict, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})
    tokenizer_load: Union[dict, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})
    tokenizer_config: Union[dict, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})
    quantization_config: Union[dict, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})
    peft_config: Union[dict, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})
    finetuner_arguments: Union[dict, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})

    dataset_response_template: Union[str, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})
    model_response_template: Union[str, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})
    system_template: Union[str, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})

    validation_split_size: Union[float, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})

    training_arguments: Union[TrainingArguments, dict, None] = field(default=None, metadata={"description": "Recipe for configuring the model."})

    num_examples: int = field(default=0, metadata={"description": "Recipe for configuring the model."})
    completion_only: bool = field(default=True, metadata={"description": "Recipe for configuring the model."})
    system_tuning: bool = field(default=False, metadata={"description": "Recipe for configuring the model."})

    verbose: bool = field(default=False, metadata={"description": "Recipe for configuring the model."})