import numpy as np
import torch
import torch.nn.functional as F
import transformers
import json
import os
from typing import Dict, Union
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from transformers import PreTrainedTokenizer, PreTrainedModel
from recipes import DatasetRecipe, YAMLDatasetRecipe, QuantizationRecipe, ModelTemplateRecipe, MistralModelTemplateRecipe, ModelRecipe, MistralModelRecipe, TokenizerRecipe, MistralTokenizerRecipe
from dispatchers import QuantizationDispatcher, ModelDispatcher, TokenizerDispatcher, DatasetDispatcher
from similarity.SimilarityAnalyzer import SimilarityAnalyzer
from utils import get_config_from_argparser
from configs import ConfigSimilarity
from dataclasses import asdict


import logging
logger = logging.getLogger("llm_steerability")
transformers.set_seed(42)

def is_jsonable(x):
    try:
        json.dumps(x); return True
    except: return False

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # ### Argparse
    # Here we create the config for training using the CLI.
    # Load the configuration instance using argparser
    config: ConfigSimilarity = get_config_from_argparser(ConfigSimilarity)
    # Set logger to debug if verbose is requested
    if config.verbose: logger.setLevel(logging.DEBUG)
    else: logger.setLevel(logging.DEBUG)
    # --------------------------------------------------------------------------

    
    # --------------------------------------------------------------------------
    # ### Create tokenizer and model
    # Here we create the tokenizer for the specific model
    tokenizer_recipe: TokenizerRecipe = config.tokenizer_recipe
    tokenizer = TokenizerDispatcher(tokenizer_recipe).get_tokenizer(config.tokenizer_name)
    # Now we create the actual model
    model_recipe: ModelRecipe = config.model_recipe
    # We also initialize the quantization config (if provided)
    quantization_recipe: QuantizationRecipe = config.quantization_recipe
    quantization_config = QuantizationDispatcher(quantization_recipe).get_quantization_config() if quantization_recipe is not None else None
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    dataset_name: str = config.dataset_name
    dataset_recipe: DatasetRecipe = config.dataset_recipe
    model_template_recipe: ModelTemplateRecipe = config.model_template_recipe
    dataset_support, dataset_test = DatasetDispatcher(dataset_recipe).get_support_and_tuning_dataset(dataset_name, 
                                                                                                      split=config.split, 
                                                                                                      num_examples=config.num_examples,
                                                                                                      postprocess_function=model_template_recipe.postprocess_function,
                                                                                                      eos_token=tokenizer.eos_token,
                                                                                                      include_labels_inside_text=False)

    similarity_analyzer = SimilarityAnalyzer(dataset_test.select(range(config.num_samples)) if config.num_samples > 0 else dataset_test,
                                             tokenizer,
                                             model_recipe=model_recipe,
                                             model_template_recipe=model_template_recipe,
                                             model_base_kwargs={"model_path": config.model_name},
                                             model_finetune_kwargs={"model_path": config.model_finetune_path},
                                             model_peft_kwargs={"model_path": config.model_name, "peft_path_or_config": config.model_peft_path},
                                             system_tuning=config.system_tuning,
                                             device=config.device)
    # --------------------------------------------------------------------------


    result = {**asdict(config), **similarity_analyzer.get_similarity_score()}
    to_pop = []
    for x in result: 
        if not is_jsonable(result[x]): to_pop.append(x)
    for x in to_pop: result.pop(x)

    logger.debug(f"Result: {result}")
    with open(os.path.join(config.output_path), "w+") as f:
        json.dump(result, f)