from transformers import PreTrainedTokenizer
from typing import Dict, Union, List, Callable
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset, DownloadMode, load_from_disk
from lm_eval.utils import load_yaml_config
from lm_eval.tasks import ConfigurableTask

# TODO: Add a preprocess function for the examples, this should also remove the need for removing labels and re-adding them

# https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt
# https://huggingface.co/docs/trl/main/en/sft_trainer
# https://www.reddit.com/r/LocalLLaMA/comments/15hz7gl/my_finetuning_based_on_llama27bchathf_model/


class DatasetRecipe:
    """
    Kwargs to give to the "load_dataset" function from "datasets" module
    """
    dataset_load_kwargs: Dict = {}

    @classmethod
    def preprocess_function(cls, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        """
        preprocess_function should be a function that returns a Dict containing:
        - "prompts": str
        - "labels": str
        so that they can be combined into:
        - "text": str
        to be used for training.
        """
        return sample

    def get_examples_lm_evaluation_harness_format(examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> str:
        """
        Given a dataset to be used for examples, returns a string containing the entries and a marker for the start and end of the examples section
        """
        if examples is None: return ""
        return "\n\n".join([ex for ex in examples["text"]]) + "\n\n"

    @classmethod
    def get_dataset(cls, 
                    dataset_path: str,
                    split: str=None, 
                    postprocess_function=None,
                    is_test: bool=False,
                    n_example: int=0,
                    **kwargs) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:

        def create_text(sample: Dict) -> Dict: 
            if is_test: sample["text"] = sample["prompts"]
            else: sample["text"] = sample["prompts"] + sample["labels"]
            return sample

        ##################################################################
        # TODO: Make an optional arg for providing an example dataset to use for dynamic selection of examples based on similarity 
        #       (right now it is the first n_examples rows of the dataset) in dataset there is probably a function that could enable such behavior     
        ##################################################################
        try: dataset = load_from_disk(dataset_path, **{**cls.dataset_load_kwargs, **kwargs})
        except: dataset = load_dataset(dataset_path, download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS, **{**cls.dataset_load_kwargs, **kwargs})

        # Split dataset if requested
        if split is not None: dataset = dataset[split]

        # If requested create a dataset of examples for prompting
        if n_example > 0: 
            dataset_len = len(dataset)
            dataset_examples = dataset.select(range(n_example))
            dataset = dataset.select(range(n_example, dataset_len))
            # Prepare the example dataset
            dataset_examples = dataset_examples.map(cls.preprocess_function, remove_columns=dataset.column_names, fn_kwargs={"cls": cls, "examples": None})
            dataset_examples = dataset_examples.map(create_text)
        else: dataset_examples = None

        # Apply the class preprocess
        dataset = dataset.map(cls.preprocess_function, remove_columns=dataset.column_names, fn_kwargs={"cls": cls, "examples": dataset_examples})

        # Join prompts and labels into "text". If we are in test mode, ignore labels for former's creation.
        dataset = dataset.map(create_text)

        # If provided, apply the postprocess_function to follow a specific prompt template
        if postprocess_function is not None: dataset = dataset.map(postprocess_function)
        return dataset

    @classmethod
    def get_response_template(cls) -> Union[str, None]: return


class LogiqaDatasetRecipe(DatasetRecipe):
    @classmethod
    def preprocess_function(cls, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> Dict:
        choices = ["a", "b", "c", "d"]
        prompt = cls.get_examples_lm_evaluation_harness_format(examples)
        prompt += "Passage: " + sample["context"] + "\n"
        prompt += "Question: " + sample["query"] + "\nChoices:\n"
        for choice, option in zip(choices, sample["options"]):
            prompt += f"{choice.upper()}. {option}\n"
        prompt += "Answer:"
        label = f'{sample["options"][int(sample["correct_option"])]}'
        return {"prompts": prompt, "labels": label}
    
    @classmethod
    def get_response_template(cls) -> Union[str, None]:
        return "\nAnswer:"


class YAMLDatasetRecipe(DatasetRecipe):
    yaml_path: str = None
    response_template: str = None
    _task: ConfigurableTask = None

    @classmethod
    def preprocess_function(cls, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        assert cls.yaml_path is not None and cls.response_template is not None, \
            "'yaml_path' and 'response_template' need to be specified. Use 'set_yaml_path' and 'set_response_template'."
        if cls._task is None: cls._task = ConfigurableTask(config=load_yaml_config(cls.yaml_path))
        prompt = (cls.get_examples_lm_evaluation_harness_format(examples) if examples is not None else "") + cls._task.fewshot_context(sample, 0)
        label = cls._task.doc_to_target(sample)
        return {"prompts": prompt, "labels": label}
    
    @classmethod
    def set_yaml_path(cls, yaml_path: str) -> None:
        cls.yaml_path = yaml_path

    @classmethod
    def set_response_template(cls, response_template: str) -> None:
        cls.response_template = response_template

    @classmethod
    def get_response_template(cls) -> Union[str, None]:
        return cls.response_template
