from typing import Dict, Union, Callable
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar


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