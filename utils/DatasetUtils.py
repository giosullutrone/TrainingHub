from typing import Dict, Union, Callable
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from transformers import PreTrainedTokenizer
from recipes import DatasetRecipe


def get_examples_lm_evaluation_harness_format(examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> str:
    """
    Given a dataset to be used for examples, returns a string containing the entries and a marker for the start and end of the examples section
    """
    if examples is None: return ""
    return "\n\n".join([ex for ex in examples["text"]]) + "\n\n"

def disable_progress_bar_wrapper(func):
    def wrapper(*args, **kwargs):
        disable_progress_bar()
        result = func(*args, **kwargs)
        enable_progress_bar()
        return result
    return wrapper

def get_response_template(tokenizer: PreTrainedTokenizer, dataset_recipe: DatasetRecipe) -> str:
    if dataset_recipe.dataset_system_message is not None: 
        conversation = [{"role": "system", "content": dataset_recipe.dataset_system_message}]
    else: conversation = []
    
    conversation_no_generation = conversation_generation = conversation + [{"role": "user", "content": "dummy"}]
    dummy_no_generation_prompt = tokenizer.apply_chat_template(
            conversation_no_generation,
            tokenize=False,
            add_generation_prompt=False,
    )

    dummy_generation_prompt = tokenizer.apply_chat_template(
            conversation_generation,
            tokenize=False,
            add_generation_prompt=True,
    )
    if len(dummy_no_generation_prompt) != len(dummy_generation_prompt):  
        return dummy_generation_prompt[len(dummy_no_generation_prompt):]
        
    conversation_no_generation = conversation + [{"role": "user", "content": "dummy"}]
    conversation_generation = conversation + [{"role": "user", "content": "ymmud"}]
    dummy_no_generation_prompt = tokenizer.apply_chat_template(
            conversation_no_generation,
            tokenize=False,
            add_generation_prompt=False,
    )
    
    dummy_generation_prompt = tokenizer.apply_chat_template(
            conversation_generation,
            tokenize=False,
            add_generation_prompt=True,
    )

    for index, (c1, c2) in enumerate(zip(dummy_no_generation_prompt[::-1], 
                                        dummy_generation_prompt[::-1])):
        if c1 != c2: return dummy_generation_prompt[-index:]
    raise Exception("An error has occured during the creation of the response template. If the problem persists, set completion_only to False.")