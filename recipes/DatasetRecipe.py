from transformers import PreTrainedTokenizer
from typing import Dict, Union, List
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset, DownloadMode
import re


# https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt
# https://huggingface.co/docs/trl/main/en/sft_trainer
# https://www.reddit.com/r/LocalLLaMA/comments/15hz7gl/my_finetuning_based_on_llama27bchathf_model/

class DatasetRecipe:
    dataset_load_kwargs: Dict = {}
    """
    preprocess_function should be a function that takes as input a Dict and returns a Dict containing:
    - "prompts": str
    - "labels": str
    so that they can be coming into "text"
    """
    preprocess_function = lambda x, examples: x

    def _get_examples(examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> str:
        if examples is None: return ""
        return "Examples:\n" + "\n".join([ex for ex in examples["text"]]) + "\nEnd of Examples\n"

    @classmethod
    def get_dataset(cls, 
                    dataset_path: str,
                    split: str=None, 
                    preprocess_function=None,
                    is_test: bool=False,
                    n_example: int=0,
                    eos_token: str="</s>",
                    **kwargs) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        def create_text(sample): sample["text"] = sample["prompts"] + sample["labels"]; return sample
        def clear_labels(sample): sample["labels"] = ""; return sample
        def restore_labels(sample, idx): sample["labels"] = _labels[idx]; return sample

        ##################################################################
        # TODO: Make an optional arg for providing an example dataset to use for dynamic selection of examples based on similarity 
        #       (right now it is the first n_examples rows of the dataset)
        #       in dataset there is probably a function that could enable such behavior     
        ##################################################################

        dataset = load_dataset(dataset_path, download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS, **{**cls.dataset_load_kwargs, **kwargs})
        # Split dataset if requested
        if split is not None: dataset = dataset[split]
        # If requested create a dataset of examples for prompting
        if n_example > 0: 
            dataset_len = len(dataset)
            dataset_examples = dataset.select(range(n_example))
            dataset = dataset.select(range(n_example, dataset_len))
            # Prepare the example dataset
            dataset_examples = dataset_examples.map(cls.preprocess_function, remove_columns=dataset.column_names, fn_kwargs={"examples": None})
            dataset_examples = dataset_examples.map(create_text)
        else: dataset_examples = None
        # Apply the class preprocess
        dataset = dataset.map(cls.preprocess_function, remove_columns=dataset.column_names, fn_kwargs={"examples": dataset_examples})
        # If is_test then save the labels for later and remove them from the dataset so that future function can't use them
        if is_test:
            _labels = dataset["labels"]
            dataset = dataset.map(clear_labels)
        # Join inputs and labels into "text" with the bos and eos token given
        dataset = dataset.map(create_text)
        # If provided, apply the preprocess function to follow a specific prompt template
        if preprocess_function is not None: dataset = dataset.map(preprocess_function)
        # If is_test since "text" has already been created without labels, restore them to be used when needed
        if is_test: 
            dataset = dataset.map(restore_labels, with_indices=True)
        # else:
        #     # TODO: to verify if sfttrainer already appends it
        #     # If not is_test then we add the eos_token to the "text"
        #     def add_eos(sample): sample["text"] += eos_token; return sample
        #     dataset = dataset.map(add_eos)
        return dataset

    @classmethod
    def get_response_template(cls) -> str: return


class LogiqaDatasetRecipe(DatasetRecipe):
    def preprocess_function(sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> Dict:
        choices = ["a", "b", "c", "d"]
        prompt = LogiqaDatasetRecipe._get_examples(examples)
        prompt += "Passage: " + sample["context"] + "\n"
        prompt += "Question: " + sample["query"] + "\nChoices:\n"
        for choice, option in zip(choices, sample["options"]):
            prompt += f"{choice.upper()}. {option}\n"
        prompt += "Answer:"
        label = f'{sample["options"][int(sample["correct_option"])]}'
        return {"prompts": prompt, "labels": label}
    
    @classmethod
    def get_response_template(cls) -> str:
        return "\nAnswer:"
    

class MathqaValueDatasetRecipe(DatasetRecipe):
    def preprocess_function(sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> Dict:
        idx = ['a', 'b', 'c', 'd', 'e'].index(sample["correct"])
        choices = [c[4:].rstrip(" ,") for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", sample["options"])]
        prompt = LogiqaDatasetRecipe._get_examples(examples)
        prompt += "Question: " + sample["Problem"] + "\nAnswer:"
        label = choices[idx]
        return {"prompts": prompt, "labels": label}
    
    @classmethod
    def get_response_template(cls) -> str:
        return "\nAnswer:"

class MathqaDatasetRecipe(DatasetRecipe):
    def preprocess_function(sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> Dict:
        prompt = LogiqaDatasetRecipe._get_examples(examples)
        prompt += "Question: " + sample["Problem"] + f"\nOptions:\n{sample['options']}" + "\nAnswer:"
        label = sample["correct"]
        return {"prompts": prompt, "labels": label}
    
    @classmethod
    def get_response_template(cls) -> str:
        return "\nAnswer:"


class GSM8KDatasetRecipe(DatasetRecipe):
    dataset_load_kwargs = {
        "name": "main"
    }

    def preprocess_function(sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> Dict:
        prompt = f'Question: {sample["question"]}\nAnswer:'
        label = {sample["answer"]}
        return {"prompts": prompt, "labels": label}
    
    @classmethod
    def get_response_template(cls) -> str:
        return "\nAnswer:"