import transformers
from transformers import TrainingArguments
from typing import Dict, Union
from finetuners import FineTuner
from recipes import DatasetRecipe, PeftRecipe, ModelRecipe, ModelTemplateRecipe, TokenizerRecipe, QuantizationRecipe
from dispatchers import DatasetDispatcher, ModelDispatcher, PeftDispatcher, QuantizationDispatcher, TokenizerDispatcher
from utils import SystemTuning, fit_response_template_tokens, fit_system_template_tokens, get_config_from_argparser, most_common_words
from configs import ConfigTrain
from peft import PromptTuningConfig

import logging
logger = logging.getLogger("llm_steerability")


transformers.set_seed(42)


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # ### Argparse
    # Here we create the config for training using the CLI.
    # Load the configuration instance using argparser
    config: ConfigTrain = get_config_from_argparser(ConfigTrain)
    # Set logger to debug if verbose is requested
    if config.verbose: logger.setLevel(logging.DEBUG)
    else: logger.setLevel(logging.INFO)
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Training Arguments definition
    # Here we create the training arguments that will be used for our training.
    # For more information, check out the huggingface documentation for "TrainingArguments"
    training_arguments: TrainingArguments = config.training_arguments
    logger.debug(f"Training Arguments used: {training_arguments}")
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # Here we define the base model we want to train.
    # Note: can be both huggingface's path or a local path
    model_name: str = config.model_name
    logger.debug(f"Model name: {model_name}")
    # An output path where to save the final model
    #   Tip: normally at the end of the training tha weights saved are the latest but they often are not
    #   the best. To solve this in the "TrainingArguments" we can specify "load_best_model_at_end=True"
    #   So that at the end of the procedure the best weights (calculated using loss on validation set) are
    #   loaded and then saved.
    #   WARNING: In some situations the trainer may not be able to load the best model at the end
    #            [In the terminal "Could not locate the best model..." or similar will appear].
    #            In that case use save_total_limit
    output_path: str = training_arguments.output_dir
    logger.debug(f"Output path: {output_path}")
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Peft config initialization
    # For the fine-tuning we can use different peft adapters. 
    # Tip: Check out the PeftRecipe for the list of recipes available or create a custom one by extending the PeftRecipe class.
    # Here we initialize the adapter's configuration used in the following sections.
    peft_recipe: PeftRecipe = config.peft_recipe
    peft_config = PeftDispatcher(peft_recipe).get_peft_config() if peft_recipe is not None else None

    if peft_recipe is not None and peft_recipe.peft_config_obj == PromptTuningConfig and config.prompt_tuning_most_common_init:
        dataset_train = DatasetDispatcher(config.dataset_recipe).get_tuning_dataset(config.dataset_name, 
                                                                                    split="train",
                                                                                    eos_token="",
                                                                                    include_labels_inside_text=False)
        words = (" ".join(list(dataset_train["text"]))).lower().split(" ")
        peft_config.prompt_tuning_init_text = " ".join(most_common_words(words, peft_config.num_virtual_tokens))
        logger.debug(f"Prompt tuning init text set to: {peft_config.prompt_tuning_init_text}")
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Create tokenizer and model
    # Here we create the tokenizer for the specific model
    tokenizer_recipe: TokenizerRecipe = config.tokenizer_recipe
    tokenizer = TokenizerDispatcher(tokenizer_recipe).get_tokenizer(model_name)
    # Now we create the actual model
    model_recipe: ModelRecipe = config.model_recipe
    # We also initialize the quantization config (if provided)
    quantization_recipe: QuantizationRecipe = config.quantization_recipe
    quantization_config = QuantizationDispatcher(quantization_recipe).get_quantization_config() if quantization_recipe is not None else None
    # Note: during training, instead of providing the peft's config, you provide the path to the fine-tuned folder that will contain (if used) the trained adapter
    model = ModelDispatcher(model_recipe).get_model(model_name, quantization_config, peft_config)
    # --------------------------------------------------------------------------
    
    outputs = model.generate(
        input_ids=tokenizer.encode("[INST] This is a test, answer with a fun fact about life: [/INST]", return_tensors="pt").to(model.device),
        generation_config=transformers.GenerationConfig(
            max_new_tokens=100,
            num_return_sequences=4,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    )
    for output in outputs:
        generated_texts = tokenizer.decode(output, skip_special_tokens=True)
        print("\n###########\n", generated_texts)