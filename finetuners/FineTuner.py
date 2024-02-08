from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import Dict, Union, Sequence, List
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset


class FineTuner:
    def __init__(self, 
                model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                dataset_train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
                dataset_validation: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]=None,
                response_template: Union[List[int], None]=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        self.response_template = response_template

    def train(self, save_path: Union[str, None]=None, training_arguments: Union[TrainingArguments, None]=None, **kwargs):
        if self.response_template is not None: data_collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer=self.tokenizer)
        else: data_collator = None

        trainer = SFTTrainer(model=self.model, 
                            tokenizer=self.tokenizer,
                            train_dataset=self.dataset_train,
                            eval_dataset=self.dataset_validation,
                            data_collator=data_collator,
                            args=training_arguments,
                            dataset_text_field="text",
                            **kwargs)
        trainer.train()
        if save_path: trainer.save_state(); trainer.save_model(save_path)
