from typing import Dict, Union
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from cookbooks import DATASET_COOKBOOK
from recipes.datasets import DatasetRecipe
import tiktoken 


@DATASET_COOKBOOK.register()
class PostOCRCorrection(DatasetRecipe):
    RESPONSE_TEMPLATE = "\nCorrected Text:"
    encoder = tiktoken.get_encoding("cl100k_base")

    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, dict, None]) -> Dict:
        prompt = f"Text: {sample['text']}\nCorrected Text:"
        label = sample["corrected_text"]
        return {"prompts": prompt[:1024], "labels": label[:1024]}