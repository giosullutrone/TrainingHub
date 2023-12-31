from transformers import PreTrainedTokenizer
from typing import Dict, Union, List, Callable
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset, DownloadMode, load_from_disk
from lm_eval.utils import load_yaml_config
from lm_eval.tasks import ConfigurableTask
from recipes.DatasetRecipe import DatasetRecipe
# TODO: Add a preprocess function for the examples, this should also remove the need for removing labels and re-adding them


# https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt
# https://huggingface.co/docs/trl/main/en/sft_trainer
# https://www.reddit.com/r/LocalLLaMA/comments/15hz7gl/my_finetuning_based_on_llama27bchathf_model/


class DatasetDispatcher:
    def __init__(self, dataset_recipe: DatasetRecipe) -> None:
        self._dataset_recipe = dataset_recipe

    def get_dataset(self, 
                    dataset_path: str,
                    split: str=None, 
                    postprocess_function=None,
                    is_test: bool=False,
                    num_examples: int=0) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:

        def create_text(sample: Dict, labels: bool) -> Dict: 
            if labels: sample["text"] = sample["prompts"] + sample["labels"]
            else: sample["text"] = sample["prompts"] 
            return sample

        try: dataset = load_from_disk(dataset_path, **self.dataset_recipe.dataset_load)
        except: dataset = load_dataset(dataset_path, download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS, **self.dataset_recipe.dataset_load)

        # Split dataset if requested
        if split is not None: dataset = dataset[split]

        # If requested create a dataset of examples for prompting
        if num_examples > 0: 
            dataset_len = len(dataset)
            dataset_examples = dataset.select(range(num_examples))
            dataset = dataset.select(range(num_examples, dataset_len))
            # Prepare the example dataset
            dataset_examples = dataset_examples.map(self.dataset_recipe.preprocess_function, remove_columns=dataset.column_names, fn_kwargs={"examples": None})
            dataset_examples = dataset_examples.map(create_text, fn_kwargs={"labels": True})
        else: dataset_examples = None

        # Apply the config preprocess
        dataset = dataset.map(self.dataset_recipe.preprocess_function, remove_columns=dataset.column_names, fn_kwargs={"examples": dataset_examples})

        # Join prompts and labels into "text". If we are in test mode, ignore labels for former's creation.
        dataset = dataset.map(create_text, fn_kwargs={"labels": is_test})

        # If provided, apply the postprocess_function to follow a specific prompt template
        if postprocess_function is not None: dataset = dataset.map(postprocess_function)
        return dataset

    @property
    def dataset_recipe(self) -> DatasetRecipe:
        return self._dataset_recipe