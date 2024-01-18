import transformers
from transformers import TrainingArguments
from typing import Dict, Union
from finetuners import FineTuner
from recipes import DatasetRecipe, PeftRecipe, ModelRecipe, ModelTemplateRecipe, TokenizerRecipe, QuantizationRecipe, LogiqaDatasetRecipe, YAMLDatasetRecipe
from dispatchers import DatasetDispatcher, ModelDispatcher, PeftDispatcher, QuantizationDispatcher, TokenizerDispatcher
from utils import SystemTuning, fit_response_template_tokens, fit_system_template_tokens, get_config_from_argparser
from configs import ConfigTrain
from peft import PromptTuningConfig
from collections import Counter

def most_common_strings(strings, n):
    # Count occurrences of each string
    string_counts = Counter(strings)

    # Get the N most common strings
    most_common = string_counts.most_common(n)

    # Extract only the strings from the result
    result = [item[0] for item in most_common]

    return result


dataset_name: str = "math_qa"
dataset_recipe: DatasetRecipe = YAMLDatasetRecipe(dataset_load={"yaml_path": "../lm-evaluation-harness-prompt-template/lm_eval/tasks/mathqa/mathqa.yaml"})
dataset_support, dataset_train = DatasetDispatcher(dataset_recipe).get_support_and_tuning_dataset(dataset_name, 
                                                                                                    split="train", 
                                                                                                    num_examples=0,
                                                                                                    eos_token="",
                                                                                                    include_labels_inside_text=False)

words = (" ".join(list(dataset_train["text"]))).lower().split(" ")
print(" ".join(most_common_strings(words, 64)))