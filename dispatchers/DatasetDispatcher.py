from typing import Dict, Union, List, Callable, Tuple
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset, DownloadMode, load_from_disk
from recipes.DatasetRecipe import DatasetRecipe


# https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt
# https://huggingface.co/docs/trl/main/en/sft_trainer
# https://www.reddit.com/r/LocalLLaMA/comments/15hz7gl/my_finetuning_based_on_llama27bchathf_model/


class DatasetDispatcher:
    def __init__(self, dataset_recipe: DatasetRecipe) -> None:
        self._dataset_recipe = dataset_recipe

    def _load_dataset(self, 
                     dataset_path: str,
                     split: str) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        try: dataset = load_from_disk(dataset_path, **self.dataset_recipe.dataset_load)
        except: dataset = load_dataset(dataset_path, download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS, **self.dataset_recipe.dataset_load)

        # Split dataset if requested
        if split is not None: dataset = dataset[split]
        return dataset

    def _process_dataset(self,
                         dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
                         dataset_support: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
                         postprocess_function,
                         eos_token: str,
                         include_labels_inside_text: bool) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:

        def create_text(sample: Dict, labels: bool) -> Dict: 
            if labels: sample["text"] = sample["prompts"] + sample["labels"] + eos_token
            else: sample["text"] = sample["prompts"] 
            return sample

        # Apply the preprocess
        dataset = dataset.map(self.dataset_recipe.preprocess_function, remove_columns=dataset.column_names, fn_kwargs={"examples": dataset_support})

        # If provided, apply the postprocess_function to follow a specific prompt template
        if postprocess_function is not None: dataset = dataset.map(postprocess_function)

        # Join prompts and labels into "text". If we are in test mode, ignore labels for former's creation.
        dataset = dataset.map(create_text, fn_kwargs={"labels": include_labels_inside_text})
        return dataset

    def get_support_and_tuning_dataset(self, 
                                       dataset_path: str, 
                                       split: str=None, 
                                       num_examples: int=0, 
                                       postprocess_function=None,
                                       eos_token: str="</s>",
                                       include_labels_inside_text: bool=True) -> Tuple[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None],
                                                                                       Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]]:
        # Load the dataset
        dataset = self._load_dataset(dataset_path, split)

        # If requested create a dataset of examples for prompting
        if num_examples > 0: 
            dataset_len = len(dataset)
            dataset_support = dataset.select(range(num_examples))
            dataset = dataset.select(range(num_examples, dataset_len))

            # Process the datasets
            dataset_support = self._process_dataset(dataset_support, 
                                                    dataset_support=None, 
                                                    postprocess_function=None,
                                                    eos_token="",
                                                    include_labels_inside_text=True)
        else: dataset_support = None

        dataset = self._process_dataset(dataset, 
                                        dataset_support=dataset_support, 
                                        postprocess_function=postprocess_function, 
                                        eos_token=eos_token,
                                        include_labels_inside_text=include_labels_inside_text)
        return dataset_support, dataset

    def get_tuning_dataset(self, 
                           dataset_path: str,
                           split: str=None, 
                           dataset_support: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]=None,
                           postprocess_function=None,
                           eos_token: str="</s>",
                           include_labels_inside_text: bool=True) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        # Load the dataset
        dataset = self._load_dataset(dataset_path, split)

        # Process the dataset
        dataset = self._process_dataset(dataset, 
                                        dataset_support=dataset_support, 
                                        postprocess_function=postprocess_function, 
                                        eos_token="",
                                        include_labels_inside_text=include_labels_inside_text)
        return dataset

    @property
    def dataset_recipe(self) -> DatasetRecipe:
        return self._dataset_recipe