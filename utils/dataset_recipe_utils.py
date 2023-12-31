from typing import Union
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset


def get_examples_lm_evaluation_harness_format(examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> str:
    """
    Given a dataset to be used for examples, returns a string containing the entries and a marker for the start and end of the examples section
    """
    if examples is None: return ""
    return "\n\n".join([ex for ex in examples["text"]]) + "\n\n"