import numpy as np
import torch
import torch.nn.functional as F
import transformers
import json
import os
from tqdm import tqdm
from typing import Dict, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from transformers import PreTrainedTokenizer, PreTrainedModel
from recipes import DatasetRecipe, YAMLDatasetRecipe, QuantizationRecipe, ModelTemplateRecipe, MistralModelTemplateRecipe, ModelRecipe, MistralModelRecipe, TokenizerRecipe, MistralTokenizerRecipe
from dispatchers import QuantizationDispatcher, ModelDispatcher, TokenizerDispatcher, DatasetDispatcher
from utils import SystemTuning, fit_response_template_tokens, fit_system_template_tokens


class SimilarityAnalyzer:
    def __init__(self,
                 dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], 
                 tokenizer: PreTrainedTokenizer, 
                 model_recipe: ModelRecipe,
                 model_template_recipe: ModelTemplateRecipe,
                 model_base_kwargs: Dict,
                 model_finetune_kwargs: Dict,
                 model_peft_kwargs: Dict,
                 system_tuning: bool,
                 device: str) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_template_recipe = model_template_recipe
        self.model_dispatcher = ModelDispatcher(model_recipe)
        self.model_base_kwargs = model_base_kwargs
        self.model_finetune_kwargs = model_finetune_kwargs
        self.model_peft_kwargs = model_peft_kwargs
        self.system_tuning = system_tuning
        self.device = device

    def get_attentions(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        attentions = []
        for input_text in tqdm(self.dataset["text"]):
            # Tokenize the input text
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            # Generate only one new token since what we are interested in are the attention output
            outputs = model.generate(input_ids=input_ids,
                                    return_dict_in_generate=True, 
                                    output_hidden_states=False,
                                    output_attentions=True,
                                    max_new_tokens=1,
                                    pad_token_id=tokenizer.pad_token_id)
            # Get the element from the batch and save the last attention layer's output
            # Note: this will have shape (n_heads, n_input_ids, n_input_ids)
            attentions.append(outputs.attentions[0][-1].float().cpu().numpy())
        return attentions

    def _compute_cosine_similarity(self, attentions_base, attentions_finetune, attentions_peft):
        # Compute the updates in attention for both finetuned and prompted methods.
        # In particular, consider the updates for the attention of the last token
        finetune_update = (attentions_finetune[..., -1, :] - attentions_base[..., -1, :])
        # For the prompted, remember that we are prepending virtual tokens so we have to ignore them
        # to match the original dimensions.
        num_virtual_tokens = attentions_peft.shape[-1] - attentions_base.shape[-1]
        peft_update = (attentions_peft[..., -1, num_virtual_tokens:] - attentions_base[..., -1, :])
        # Calculate the cosine similarity on the flattened matrices.
        # Note: we can safely flatten them since each row contains the attention for a specific attention head
        similarity = cosine_similarity(finetune_update.reshape(1, -1), peft_update.reshape(1, -1))[0, 0]
        return similarity
    
    def _compute_euclidean_similarity(self, attentions_base, attentions_finetune, attentions_peft):
        base_distance = euclidean_distances(attentions_finetune[..., -1, :].reshape(1, -1), attentions_base[..., -1, :].reshape(1, -1))
        num_virtual_tokens = attentions_peft.shape[-1] - attentions_base.shape[-1]
        peft_distance = euclidean_distances(attentions_finetune[..., -1, :].reshape(1, -1), attentions_peft[..., -1, num_virtual_tokens:].reshape(1, -1))
        similarity = (base_distance - peft_distance) / base_distance
        return similarity

    def get_similarity_score(self) -> Dict:
        model_base = self.model_dispatcher.get_model(**self.model_base_kwargs).to(self.device)
        attentions_base = self.get_attentions(model_base, self.tokenizer)
        model_base = None

        model_finetune = self.model_dispatcher.get_model(**self.model_finetune_kwargs).to(self.device)
        attentions_finetune = self.get_attentions(model_finetune, self.tokenizer)
        model_finetune = None

        model_peft = self.model_dispatcher.get_model(**self.model_peft_kwargs).to(self.device)
        if self.system_tuning:
            system_template = fit_system_template_tokens(self.dataset, self.model_template_recipe, self.tokenizer)
            model_peft = SystemTuning.add_proxy(model_peft, system_template=system_template)
        attentions_peft = self.get_attentions(model_peft, self.tokenizer)
        model_peft = None

        similarities_cosine = []
        similarities_euclidean = []
        # For each element in the attentions gathered (i.e. for each prompt in the dataset)
        for base, finetuned, prompted in tqdm(zip(attentions_base, attentions_finetune, attentions_peft)):
            similarities_cosine.append(self._compute_cosine_similarity(base, finetuned, prompted))
        for base, finetuned, prompted in tqdm(zip(attentions_base, attentions_finetune, attentions_peft)):
            similarities_euclidean.append(self._compute_euclidean_similarity(base, finetuned, prompted))
        result = {
            "cosine": float(np.average(similarities_cosine)),
            "euclidean": float(np.average(similarities_euclidean))
        }
        return result
