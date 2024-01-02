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
                 preprocess_function: Callable[[Dict, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]], Dict]=None, 
                 dataset_load: Union[Dict, None]=None,
                 response_template: Union[str, None]=None) -> None:
        if preprocess_function is not None: self.preprocess_function = preprocess_function
        self._dataset_load = {}
        if self.DATASET_LOAD is not None: self._dataset_load.update(self.DATASET_LOAD)
        if dataset_load is not None: self._dataset_load.update(dataset_load)
        self._response_template = response_template if response_template else self.RESPONSE_TEMPLATE

    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        return sample

    @property
    def dataset_load(self) -> Dict:
        return self._dataset_load
    
    @property
    def response_template(self) -> Union[str, None]:
        return self._response_template


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
class YAMLDatasetRecipe(DatasetRecipe):
    def __init__(self, 
                 yaml_path: str,
                 response_template: Union[str, None]=None,
                 preprocess_function: Callable[[Dict, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]], Dict]=None) -> None:
        super().__init__(preprocess_function, response_template)
        self.yaml_path = yaml_path
        self._task = ConfigurableTask(config=load_yaml_config(self.yaml_path))

    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        prompt = (get_examples_lm_evaluation_harness_format(examples) if examples is not None else "") + self._task.fewshot_context(sample, 0)
        label = self._task.doc_to_target(sample)
        if isinstance(label, int): 
            # If the label is an integer and the task is a multiple choice one then it is the index of the element in doc_to_choice
            # Check out: https://github.com/giosullutrone/lm-evaluation-harness-prompt-template/blob/main/lm_eval/api/task.py
            if self._task.OUTPUT_TYPE == "multiple_choice": label = f"{self._task.config.target_delimiter}{self._task.doc_to_choice(sample)[label]}"
            else: label = str(label)
        return {"prompts": prompt, "labels": label}
