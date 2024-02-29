import transformers
from transformers import TrainingArguments
from typing import Dict, Union
from finetuners import FineTuner
from recipes import DatasetRecipe, PeftRecipe, ModelRecipe, ModelTemplateRecipe, TokenizerRecipe, QuantizationRecipe, get_examples_lm_evaluation_harness_format, GenerationDatasetRecipe
from dispatchers import DatasetDispatcher, ModelDispatcher, PeftDispatcher, QuantizationDispatcher, TokenizerDispatcher
from utils import SystemTuning, fit_response_template_tokens, fit_system_template_tokens, get_config_from_argparser, most_common_words
from configs import ConfigGenerateDataset
from peft import PromptTuningConfig

import logging
logger = logging.getLogger("llm_steerability")

transformers.set_seed(42)


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # ### Argparse
    # Here we create the config for training using the CLI.
    # Load the configuration instance using argparser
    config: ConfigGenerateDataset = get_config_from_argparser(ConfigGenerateDataset)
    # Set logger to debug if verbose is requested
    if config.verbose: logger.setLevel(logging.DEBUG)
    else: logger.setLevel(logging.INFO)
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # Here we define the base model we want to train.
    # Note: can be both huggingface's path or a local path
    model_name: str = config.generation_model_name
    logger.debug(f"Model name: {model_name}")
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Peft config initialization
    # For the fine-tuning we can use different peft adapters. 
    # Tip: Check out the PeftRecipe for the list of recipes available or create a custom one by extending the PeftRecipe class.
    # Here we initialize the adapter's configuration used in the following sections.
    peft_recipe: PeftRecipe = config.generation_peft_recipe
    peft_config = PeftDispatcher(peft_recipe).get_peft_config() if peft_recipe is not None else None
    if config.generation_peft_path is not None: peft_config = config.generation_peft_path
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Create tokenizer and model
    # Here we create the tokenizer for the specific model
    tokenizer_recipe: TokenizerRecipe = config.generation_tokenizer_recipe
    tokenizer = TokenizerDispatcher(tokenizer_recipe).get_tokenizer(model_name)
    config.generation_config.pad_token_id = tokenizer.pad_token_id
    # Now we create the actual model
    model_recipe: ModelRecipe = config.generation_model_recipe
    # We also initialize the quantization config (if provided)
    quantization_recipe: QuantizationRecipe = config.generation_quantization_recipe
    quantization_config = QuantizationDispatcher(quantization_recipe).get_quantization_config() if quantization_recipe is not None else None
    # Note: during training, instead of providing the peft's config, you provide the path to the fine-tuned folder that will contain (if used) the trained adapter
    model = ModelDispatcher(model_recipe).get_model(model_name, quantization_config, peft_config)
    # --------------------------------------------------------------------------
    
    
    # --------------------------------------------------------------------------
    # ### Dataset
    # We load the datasets from huggingface using the specific recipe.
    # Tip: For more datasets check out recipes.DatasetRecipe, if you don't find the one that you want
    # follow the dataset recipe creation guide. 
    # TODO: Write a guide for the creation of new recipes
    # Tip: You can also use the YAML recipe to re-use any yaml config file available in lm-evaluation-harness library.
    #
    # Here we are creating a train and validation datasets with the given number of examples per prompt and
    # adapted to the template of the specific model:
    # ### Example of a prompt:
    # "What is the result of 2 + 2? Answer: "
    # -> for LLama2 we obtain "<SYS> </SYS> [INST] What is the result of 2 + 2? Answer: [/INST]"
    dataset_name: str = config.generation_dataset_name
    dataset_recipe: GenerationDatasetRecipe = config.generation_dataset_recipe
    model_template_recipe: ModelTemplateRecipe = config.generation_model_template_recipe
    _, dataset_generation = DatasetDispatcher(dataset_recipe).get_support_and_tuning_dataset(dataset_name, 
                                                                                            split="train", 
                                                                                            num_examples=config.generation_num_examples,
                                                                                            postprocess_function=model_template_recipe.postprocess_function,
                                                                                            eos_token="",
                                                                                            include_labels_inside_text=True,
                                                                                            num_proc=config.num_proc,
                                                                                            dynamic_examples=True)

    if config.generation_starting_size: dataset_generation = dataset_generation.select(range(config.generation_starting_size))

    # Logger
    logger.debug(f'First prompt for the generation set:\n{dataset_generation["text"][0]}')
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Peft config initialization
    # For the fine-tuning we can use different peft adapters. 
    # Tip: Check out the PeftRecipe for the list of recipes available or create a custom one by extending the PeftRecipe class.
    # Here we initialize the adapter's configuration used in the following sections.
    peft_recipe: PeftRecipe = config.generation_peft_config
    peft_config = PeftDispatcher(peft_recipe).get_peft_config() if peft_recipe is not None else None

    if config.generation_peft_path is not None: peft_config = config.generation_peft_path
    # --------------------------------------------------------------------------
    
    def sample(sample):
        splitter = model_template_recipe.model_response_template if model_template_recipe.model_response_template is not None else dataset_recipe.dataset_response_template
        texts = model.generate(input_ids=tokenizer.encode(sample["text"][0], return_tensors="pt").to(model.device), generation_config=config.generation_config)
        texts = [tokenizer.decode(x, skip_special_tokens=True).split(splitter)[-1] for x in texts]
        return {"text": texts}

    dataset_generated = dataset_generation.map(sample, desc="Generating new samples", remove_columns=dataset_generation.column_names, batched=True, batch_size=1)
    dataset_generated = dataset_generated.map(dataset_recipe.postprocess_generation_function, desc="Applying post-process to generated dataset")
    dataset_generated.save_to_disk(config.generation_output_path)
    logger.debug(f'First prompt for the generated set:\n{dataset_generated["text"][0]}')

    