from typing import Dict, Union, Callable
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from lm_eval.tasks import ConfigurableTask
from lm_eval.utils import load_yaml_config
from cookbooks import DATASET_COOKBOOK
from utils.DatasetUtils import get_examples_lm_evaluation_harness_format
from recipes.datasets import DatasetRecipe


@DATASET_COOKBOOK.register()
class YAMLDatasetRecipe(DatasetRecipe):
    def __init__(self, 
                 preprocess_dataset: Callable[[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]], Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]]=None, 
                 preprocess_function: Callable[[Dict, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, dict, None]], Dict]=None, 
                 dataset_load: Union[Dict, None]=None,
                 dataset_response_template: Union[str, None]=None) -> None:
        self.yaml_path = dataset_load.pop("yaml_path")
        self._task = ConfigurableTask(config=load_yaml_config(self.yaml_path))
        if self._task._config.process_docs and not preprocess_dataset: preprocess_dataset = self._task._config.process_docs
        super().__init__(preprocess_dataset, preprocess_function, dataset_load, dataset_response_template)

    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, dict, None]) -> Dict:
        try: prompt = (get_examples_lm_evaluation_harness_format(examples) if examples is not None else "") + self._task.fewshot_context(sample, 0)
        except Exception as e: print(e); return {"prompts": None, "labels": None}
        label = self._task.doc_to_target(sample)
        if isinstance(label, int): 
            # If the label is an integer and the task is a multiple choice one then it is the index of the element in doc_to_choice
            # Check out: https://github.com/giosullutrone/lm-evaluation-harness-prompt-template/blob/main/lm_eval/api/task.py
            if self._task.OUTPUT_TYPE == "multiple_choice": label = f"{self._task.config.target_delimiter}{self._task.doc_to_choice(sample)[label]}"
        return {"prompts": str(prompt), "labels": str(label)}