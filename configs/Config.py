from typing import Dict, Union
from transformers import TrainingArguments
from recipes.DatasetRecipe import DatasetRecipe
from recipes.ModelRecipe import ModelRecipe
from recipes.TokenizerRecipe import TokenizerRecipe
from recipes.QuantizationRecipe import QuantizationRecipe
from recipes.ModelTemplateRecipe import ModelTemplateRecipe
from recipes.PeftRecipe import PeftRecipe
from dataclasses import dataclass, field


@dataclass
class Config:
    # TODO: add more info
    """
    Configuration class for training by CLI.

    Attributes:
        - `dataset_recipe` (DatasetRecipe): Recipe for loading and preprocessing the dataset.
        - `model_recipe` (ModelRecipe): Recipe for configuring the model.
        - `model_template_recipe` (ModelTemplateRecipe): Recipe for configuring the model's template.
        - `tokenizer_recipe` (TokenizerRecipe): Recipe for configuring the tokenizer.
        - `quantization_recipe` (Union[QuantizationRecipe, None]): Recipe for model quantization (if applicable).
        - `peft_recipe` (Union[PeftRecipe, None]): Recipe for performance enhancement and fine-tuning (if applicable).
        - `dataset_name` (Union[str, None]): Name or path of the dataset. If not specified here, can be also given by terminal.
        - `model_name` (Union[str, None]): Name or path of the model. If not specified here, can be also given by terminal.
        - `validation_split_size` (Union[float, None]): Size of the validation split as a fraction of the dataset size. Used only if validation set doesn't exist for the dataset.
        - `num_examples` (int): Number of examples to give each prompt. They will be taken from the first elements num_examples elements of the train set. Default: 0.
        - `training_arguments` (Union[TrainingArguments, None]): Arguments for training the model (if applicable).
        - `finetuner_arguments` (Union[Dict, None]): Arguments for fine-tuning the model (if applicable).
        - `completion_only` (bool): Whether to train on prompt completion only or to predict the whole prompt. Default: True.
        - `system_tuning` (bool): Whether to train `prompt tuning` on system prompts instead of the naive prepending. Only applicable if using `prompt tuning` as peft Default: False.
    """
    dataset_recipe: DatasetRecipe
    model_recipe: ModelRecipe
    model_template_recipe: ModelTemplateRecipe
    tokenizer_recipe: TokenizerRecipe
    quantization_recipe: Union[QuantizationRecipe, None] = field(default=None)
    peft_recipe: Union[PeftRecipe, None] = field(default=None)
    
    dataset_name: Union[str, None] = field(default=None)
    model_name: Union[str, None] = field(default=None)
    validation_split_size: Union[float, None] = field(default=None)
    num_examples: int = field(default=0)

    training_arguments: Union[TrainingArguments, None] = field(default=None)
    finetuner_arguments: Union[Dict, None] = field(default_factory=dict)
    completion_only: bool = field(default=True)
    system_tuning: bool = field(default=False)