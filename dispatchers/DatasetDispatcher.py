from typing import Dict, Union, Tuple, Optional, List
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset, DownloadMode, load_from_disk
from recipes.datasets import DatasetRecipe
from recipes.tokenizers import TokenizerRecipe
from utils.DatasetUtils import disable_progress_bar_wrapper
from functools import partial
import random
from transformers import PreTrainedTokenizer


class DatasetDispatcher:
    def __init__(self, dataset_recipe: DatasetRecipe, tokenizer_recipe: TokenizerRecipe) -> None:
        self._dataset_recipe = dataset_recipe
        self._tokenizer_recipe = tokenizer_recipe

    def _load_dataset(self, 
                     dataset_path: str,
                     split: Optional[str],
                     dataset_size: Optional[int]) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        try: 
            dataset = load_from_disk(dataset_path, **self.dataset_recipe.dataset_load)
            # Split dataset if requested
            if split is not None: dataset = dataset[split]
        except: 
            assert split is not None, "Trying to load from hub but no split has been defined"
            dataset = load_dataset(dataset_path, download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS, **self.dataset_recipe.dataset_load)

        if dataset_size: dataset = dataset.select(range(dataset_size))
        if self.dataset_recipe.preprocess_dataset: dataset = self.dataset_recipe.preprocess_dataset(dataset)
        return dataset

    def get_dataset(
            self, 
            dataset_path: str, 
            tokenizer: PreTrainedTokenizer,
            split: Optional[str], 
            dataset_size: Optional[int],
            num_examples: int, 
            shuffle: bool, 
            shuffle_seed: int=42, 
            num_proc: Optional[int]=None, 
        ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
            assert not num_examples < 0
        
            # Load the dataset
            dataset = self._load_dataset(dataset_path, split, dataset_size)
            
            # Shuffle the dataset if requested
            if shuffle: dataset.shuffle(shuffle_seed)
            
            dataset = dataset.map(self.dataset_recipe.preprocess_function, 
                                remove_columns=dataset.column_names, 
                                num_proc=num_proc, desc="Preprocessing dataset")
            
            # Filter incorrect rows
            dataset = dataset.filter(lambda x: x["prompts"] is not None and x["labels"] is not None)

            dataset_support = dataset.copy()
            
            def get_random_examples(current_index: int) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
                indices = list(range(len(dataset)))
                indices.remove(current_index)
                return dataset_support.select(random.sample(indices, num_examples))
            
            dataset = dataset.map(lambda x, index: self.dataset_recipe.preprocess_function(x, examples=get_random_examples(index)), 
                                remove_columns=dataset.column_names, with_indices=True, 
                                num_proc=num_proc, desc="Preprocessing dataset")
            
            def postprocess(x: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> Dict:
                if self._tokenizer_recipe.tokenizer_system is not None: 
                    conversation = [{"role": "system", "content": self._tokenizer_recipe.tokenizer_system}]
                else: conversation = []
                conversation += [
                    {"role": "user", "content": x["prompts"]},
                    {"role": "assistant", "content": x["labels"]}
                ]
                
                text = tokenizer.apply_chat_template(conversation, 
                                                     chat_template=self._tokenizer_recipe.tokenizer_chat_template, 
                                                     add_generation_prompt=False,
                                                     tokenize=False)
                return {"text": text}
            
            dataset = dataset.map(postprocess, 
                                remove_columns=dataset.column_names, 
                                num_proc=num_proc, desc="Preprocessing dataset")
            
            return dataset

    @property
    def dataset_recipe(self) -> DatasetRecipe:
        return self._dataset_recipe
