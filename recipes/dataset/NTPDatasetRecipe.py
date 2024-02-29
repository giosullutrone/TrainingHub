import random
from typing import Dict, Union, Callable
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from lm_eval.tasks import ConfigurableTask
from lm_eval.utils import load_yaml_config
from cookbooks import DATASET_COOKBOOK
from .DatasetRecipe import DatasetRecipe, get_examples_lm_evaluation_harness_format


@DATASET_COOKBOOK.register()
class MathqaDatasetRecipe(DatasetRecipe):
    DATASET_RESPONSE_TEMPLATE = "\nAnswer:"

    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        labels = ['a', 'b', 'c', 'd', 'e']
        def _preprocess_prompt(s: dict):
            return f"Question: {s['Problem']}\nOptions: {s['options']}\nAnswer:"
        def _preprocess_label(s: dict):
            return str(labels[labels.index(s["correct"])])
        if examples:
            examples = {"text": [_preprocess_prompt(x) + _preprocess_label(x) for x in examples]}
            prompt = get_examples_lm_evaluation_harness_format(examples)
        else: prompt = ""
        prompt += _preprocess_prompt(sample)
        return {"prompts": prompt, "labels": _preprocess_label(sample)}

@DATASET_COOKBOOK.register()
class WikipediaDatasetRecipe(DatasetRecipe):
    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        def corrupt_text(text, corruption_level):
            corrupted_text = ""
            for char in text:
                if random.random() < corruption_level:
                # Replace with a random character
                    replacement = chr(random.randint(ord('a'), ord('z')))
                    corrupted_text += replacement
                else:
                    corrupted_text += char
            return corrupted_text
        def _preprocess_prompt(s: dict):
            return corrupt_text(s['text'], random.random() / 10)
        if examples:
            examples = {"text": [_preprocess_prompt(x) + x["text"] for x in examples]}
            prompt = get_examples_lm_evaluation_harness_format(examples)
        else: prompt = ""
        prompt += _preprocess_prompt(sample)
        return {"prompts": prompt, "labels": sample["text"]}

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

@DATASET_COOKBOOK.register()
class NewsIFTDatasetRecipe(DatasetRecipe):
    def preprocess_function(self, sample, examples) -> Dict:
        classes = [
            'alt.atheism',
            'comp.graphics',
            'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware',
            'comp.windows.x',
            'misc.forsale',
            'rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey',
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space',
            'soc.religion.christian',
            'talk.politics.guns',
            'talk.politics.mideast',
            'talk.politics.misc',
            'talk.religion.misc'
        ]
        
        def _preprocess_prompt(s: dict):
            return f"News:\n{s['text']}\nTopic:"
        
        prompt = "Choose the topic for the given news among the topics provided.\nTopics: " + ", ".join(classes) + "\n\n"

        if examples:
            examples = {"text": [_preprocess_prompt(x) + x["label_text"] for x in examples]}
            prompt += get_examples_lm_evaluation_harness_format(examples)

        prompt += _preprocess_prompt(sample)
        return {"prompts": prompt, "labels": sample['label_text']}


@DATASET_COOKBOOK.register()
class NewsEFTDatasetRecipe(DatasetRecipe):
    def preprocess_function(self, sample, examples) -> Dict:
        def _preprocess_prompt(s: dict):
            return f"News:\n{s['news']}\nTopic:{s['topic']}"
        def _preprocess_label(s: dict):
            return f"\nExplanation:\n{s['text']}\nScore:{s['score']}"
        
        prompt = "You are an AI specialized in evaluating the results from a topic modeling algorithm. \
Generate a single step-by-step explanation on why the combination of news and topic have achieved a score from 0 to 5. \
Where 5 means that the algorithm has given the correct topic and 0 that the topic is the worst among the possible one.\
At the end of the explanation, provide the score."

        if examples:
            examples = {"text": [_preprocess_prompt(x) + _preprocess_label(x) for x in examples]}
            prompt += get_examples_lm_evaluation_harness_format(examples)

        prompt += _preprocess_prompt(sample)
        return {"prompts": prompt, "labels": _preprocess_label(sample)}
    
