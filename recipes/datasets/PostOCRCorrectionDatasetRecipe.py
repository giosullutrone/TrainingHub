from typing import Dict, Union
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from cookbooks import DATASET_COOKBOOK
from recipes.datasets import DatasetRecipe
import tiktoken


@DATASET_COOKBOOK.register()
class PostOCRCorrectionDatasetRecipe(DatasetRecipe):
    tik = tiktoken.get_encoding("cl100k_base")
    
    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, dict, None]) -> Dict:        
        prompt = f"Text: {sample['text'][:int(1024 * 3)]}\nCorrected Text:"
        label = sample["corrected_text"]
        return {"prompts": prompt, "labels": label[:int(1024 * 3)]}
