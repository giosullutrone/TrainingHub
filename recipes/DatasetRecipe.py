from typing import Dict, Union, Callable
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from lm_eval.tasks import ConfigurableTask
from lm_eval.utils import load_yaml_config
from cookbooks import DATASET_COOKBOOK


def get_examples_lm_evaluation_harness_format(examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> str:
    """
    Given a dataset to be used for examples, returns a string containing the entries and a marker for the start and end of the examples section
    """
    if examples is None: return ""
    return "\n\n".join([ex for ex in examples["text"]]) + "\n\n"


class DatasetRecipe:
    """Kwargs to give to the "load_dataset" function from "datasets" module"""
    DATASET_LOAD: Union[Dict, None] = None
    RESPONSE_TEMPLATE: Union[str, None] = None

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
        self._dataset_response_template = dataset_response_template if dataset_response_template else self.RESPONSE_TEMPLATE

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

@DATASET_COOKBOOK.register()
class LogiqaDatasetRecipe(DatasetRecipe):
    RESPONSE_TEMPLATE = "\nAnswer:"

    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        choices = ["a", "b", "c", "d"]
        prompt = get_examples_lm_evaluation_harness_format(examples)
        prompt += "Passage: " + sample["context"] + "\n"
        prompt += "Question: " + sample["query"] + "\nChoices:\n"
        for choice, option in zip(choices, sample["options"]):
            prompt += f"{choice.upper()}. {option}\n"
        prompt += "Answer:"
        label = f' {sample["options"][int(sample["correct_option"])]}'
        return {"prompts": prompt, "labels": label}
    
@DATASET_COOKBOOK.register()
class MathqaDatasetRecipe(DatasetRecipe):
    RESPONSE_TEMPLATE = "\nAnswer:"

    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        prompt = get_examples_lm_evaluation_harness_format(examples)
        prompt += f"Question: {sample['Problem']}\nOptions: {sample['options']}\nAnswer:"
        label = {['a', 'b', 'c', 'd', 'e'].index(sample["correct"])}
        return {"prompts": prompt, "labels": label}

@DATASET_COOKBOOK.register()
class YAMLDatasetRecipe(DatasetRecipe):
    def __init__(self, 
                 preprocess_dataset: Callable[[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]], Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]]=None, 
                 preprocess_function: Callable[[Dict, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]], Dict]=None, 
                 dataset_load: Union[Dict, None]=None,
                 dataset_response_template: Union[str, None]=None) -> None:
        self.yaml_path = dataset_load.pop("yaml_path")
        self._task = ConfigurableTask(config=load_yaml_config(self.yaml_path))
        if self._task._config.process_docs and not preprocess_dataset: preprocess_dataset = self._task._config.process_docs
        super().__init__(preprocess_dataset, preprocess_function, dataset_load, dataset_response_template)

    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        try: prompt = (get_examples_lm_evaluation_harness_format(examples) if examples is not None else "") + self._task.fewshot_context(sample, 0)
        except Exception as e: print(e); return {"prompts": None, "labels": None}
        label = self._task.doc_to_target(sample)
        if isinstance(label, int): 
            # If the label is an integer and the task is a multiple choice one then it is the index of the element in doc_to_choice
            # Check out: https://github.com/giosullutrone/lm-evaluation-harness-prompt-template/blob/main/lm_eval/api/task.py
            if self._task.OUTPUT_TYPE == "multiple_choice": label = f"{self._task.config.target_delimiter}{self._task.doc_to_choice(sample)[label]}"
        return {"prompts": str(prompt), "labels": str(label)}
