import random
from typing import Dict, Union, Callable
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from lm_eval.tasks import ConfigurableTask
from lm_eval.utils import load_yaml_config
from cookbooks import DATASET_COOKBOOK


def get_examples_lm_evaluation_harness_format(examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None], col: str="text") -> str:
    """
    Given a dataset to be used for examples, returns a string containing the entries and a marker for the start and end of the examples section
    """
    if examples is None: return ""
    return "\n\n".join([ex for ex in examples[col]]) + "\n\n"


class DatasetRecipe:
    """Kwargs to give to the "load_dataset" function from "datasets" module"""
    DATASET_LOAD: Union[Dict, None] = None
    DATASET_RESPONSE_TEMPLATE: Union[str, None] = None

    def __init__(self, 
                 preprocess_dataset: Callable[[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]], Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]]=None, 
                 preprocess_function: Callable[[Dict, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]], Dict]=None, 
                 dataset_load: Union[Dict, None]=None,
                 dataset_response_template: Union[str, None]=None) -> None:
        if preprocess_dataset is not None: self.preprocess_dataset = preprocess_dataset
        if preprocess_function is not None: self.preprocess_function = preprocess_function
        self._dataset_load = {}
        if self.DATASET_LOAD is not None: self._dataset_load.update(self.DATASET_LOAD)
        if dataset_load is not None: self._dataset_load.update(dataset_load)
        self._dataset_response_template = dataset_response_template if dataset_response_template else self.DATASET_RESPONSE_TEMPLATE

    def preprocess_dataset(self, dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        return dataset
    
    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        return sample

    @property
    def dataset_load(self) -> Dict:
        return self._dataset_load
    
    @property
    def dataset_response_template(self) -> Union[str, None]:
        return self._dataset_response_template


@DATASET_COOKBOOK.register()
class DefaultDatasetRecipe(DatasetRecipe): pass
