from datasets import load_dataset
from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import Dict, Union, Sequence, List
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from dataclasses import dataclass
import torch
import transformers
from transformers import Trainer
import numpy as np

# https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt


class FineTuner:
    def __init__(self, 
                model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                dataset_train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
                dataset_validation: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]=None,
                response_template: Union[List, None]=None,
                device_map: Union[str, Dict]={'': 0}) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        self.response_template = response_template
        self.device_map = device_map

    def train(self, output_dir: str, training_arguments: TrainingArguments=None, **kwargs):
        data_collator = DataCollatorForCompletionOnlyLM(self.response_template.tolist(), tokenizer=self.tokenizer)

        trainer = SFTTrainer(model=self.model, 
                            tokenizer=self.tokenizer,
                            train_dataset=self.dataset_train,
                            eval_dataset=self.dataset_validation,
                            data_collator=data_collator,
                            args=training_arguments,
                            **kwargs)
        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir)
