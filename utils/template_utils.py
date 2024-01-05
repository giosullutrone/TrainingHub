from typing import Union, List
import torch
from transformers import PreTrainedTokenizer
from recipes import ModelTemplateRecipe, DatasetRecipe
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from tqdm import tqdm


def get_template_token_position(x: torch.Tensor, token_ids: torch.Tensor) -> Union[int, None]:
    token_ids_start_idx = None
    for idx in torch.where(x == token_ids[0])[0]:
        # Here we are just making sure that the token IDs match
        if (idx + len(token_ids) < len(x)) and (all(token_ids == x[idx : idx + len(token_ids)])):
            token_ids_start_idx = idx
    if token_ids_start_idx is None: return
    print(len(x), token_ids_start_idx, token_ids, x)
    return int(token_ids_start_idx + len(token_ids))


def fit_template_tokens(dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], template: str, tokenizer: PreTrainedTokenizer) -> Union[List[int], None]:
    print(template)
    def _fit_template_tokens(sample: str, template: str, tokenizer: PreTrainedTokenizer):
        sample = tokenizer.encode(sample, return_tensors="pt", add_special_tokens=True)[0]
        token_ids = tokenizer.encode(template, return_tensors="pt", add_special_tokens=False)[0]
        token_ids_start_idx = None
        idx = 0
        while idx < len(token_ids):
            token_ids_start_idx = get_template_token_position(sample, token_ids[idx:])
            if token_ids_start_idx is not None: break
            idx += 1
        else: raise Exception("Could not find fitting template tokens")
        return token_ids[idx:]
    token_ids = None
    for sample in tqdm(dataset["text"], desc="Fitting template tokens"):
        _token_ids = _fit_template_tokens(sample, template, tokenizer)
        if token_ids is None or len(token_ids) > len(_token_ids): token_ids = _token_ids
        elif len(token_ids) == len(_token_ids): assert torch.equal(token_ids, _token_ids), "Found different token ids for same length"
    return token_ids.tolist()

def fit_response_template_tokens(dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], dataset_recipe: DatasetRecipe, model_template_recipe: ModelTemplateRecipe, tokenizer: PreTrainedTokenizer) -> Union[List[int], None]:
    if model_template_recipe.model_response_template: return fit_template_tokens(dataset, model_template_recipe.model_response_template, tokenizer)
    return fit_template_tokens(dataset, dataset_recipe.dataset_response_template, tokenizer)

def fit_system_template_tokens(dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], model_template_recipe: ModelTemplateRecipe, tokenizer: PreTrainedTokenizer) -> Union[List[int], None]:
    return fit_template_tokens(dataset, model_template_recipe.system_template, tokenizer)
