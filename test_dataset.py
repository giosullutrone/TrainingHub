import transformers
from transformers import TrainingArguments
from typing import Dict, Union
from finetuners import FineTuner
from recipes import DatasetRecipe, PeftRecipe, ModelRecipe, ModelTemplateRecipe, TokenizerRecipe, QuantizationRecipe, MathqaDatasetRecipe, MistralModelTemplateRecipe
from dispatchers import DatasetDispatcher, ModelDispatcher, PeftDispatcher, QuantizationDispatcher, TokenizerDispatcher
from utils import SystemTuning, fit_response_template_tokens, fit_system_template_tokens, get_config_from_argparser, most_common_words
from configs import ConfigTrain
from peft import PromptTuningConfig


import logging
logger = logging.getLogger("llm_steerability")


transformers.set_seed(42)


if __name__ == "__main__":
    dataset_name: str = "math_qa"
    dataset_recipe: DatasetRecipe = MathqaDatasetRecipe()
    model_template_recipe: ModelTemplateRecipe = MistralModelTemplateRecipe()
    dataset_support, dataset_train = DatasetDispatcher(dataset_recipe).get_support_and_tuning_dataset(dataset_name, 
                                                                                                      split="train", 
                                                                                                      num_examples=5,
                                                                                                      postprocess_function=model_template_recipe.postprocess_function,
                                                                                                      eos_token="</s>",
                                                                                                      examples_dynamic=True,
                                                                                                      include_labels_inside_text=True)
    
    # Logger
    print(f'First prompt for training set:\n{"#############".join(dataset_train["text"][0:2])}')