from typing import Dict, Union
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from cookbooks import DATASET_COOKBOOK
from utils.DatasetUtils import get_examples_lm_evaluation_harness_format
from recipes.datasets import DatasetRecipe


@DATASET_COOKBOOK.register()
class MathqaDatasetRecipe(DatasetRecipe):
    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, dict, None]) -> Dict:
        prompt = get_examples_lm_evaluation_harness_format(examples)
        prompt += f"Question: {sample['Problem']}\nOptions: {sample['options']}\nAnswer:"
        label = str(['a', 'b', 'c', 'd', 'e'].index(sample["correct"]))
        return {"prompts": prompt, "labels": label}