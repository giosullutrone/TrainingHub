from typing import Union, Optional, Dict
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from recipes.Recipe import Recipe
from cookbooks import DATASET_COOKBOOK
from dataclasses import dataclass, field


@dataclass
class DatasetRecipe(Recipe):
    """Kwargs to give to the "load_dataset" function from "datasets" module"""
    dataset_load: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "description": "Kwargs for dataset load."
        }
    )
    dataset_system_message: Optional[str] = field(
        default=None, 
        metadata={
            "description": "System messsage to use."
        }
    )

    def preprocess_dataset(self, dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        return dataset
    
    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, dict, None]) -> Dict:
        return sample

@DATASET_COOKBOOK.register()
class DefaultDatasetRecipe(DatasetRecipe): pass
