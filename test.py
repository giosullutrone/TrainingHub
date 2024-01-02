import transformers
from transformers import TrainingArguments
from typing import Dict, Union
from finetuners import FineTuner
from recipes import DatasetRecipe, PeftRecipe, ModelRecipe, ModelTemplateRecipe, TokenizerRecipe, QuantizationRecipe
from dispatchers import DatasetDispatcher, ModelDispatcher, PeftDispatcher, QuantizationDispatcher, TokenizerDispatcher
from utils import SystemTuning, fit_response_template_tokens, fit_system_template_tokens, get_config_from_argparser
from configs.Config import Config
from peft import PromptTuningConfig

import logging
logger = logging.getLogger("llm_steerability")


transformers.set_seed(42)


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # ### Argparse
    # Here we create the config for training using the CLI.
    # Load the configuration instance using argparser
    config: Config = get_config_from_argparser()
    # Set logger to debug if verbose is requested
    if config.verbose: logger.setLevel(logging.DEBUG)
    else: logger.setLevel(logging.DEBUG)
    # --------------------------------------------------------------------------

    print(config.training_arguments)