import argparse
import importlib.util
import transformers
from transformers import TrainingArguments
from typing import Dict, Union
from finetuner import FineTuner
from recipes import DatasetRecipe, PeftRecipe, ModelRecipe, ModelTemplateRecipe, TokenizerRecipe, QuantizationRecipe
from dispatchers import DatasetDispatcher, ModelDispatcher, PeftDispatcher, QuantizationDispatcher, TokenizerDispatcher
from utils import SystemTuning, fit_response_template_tokens, fit_system_template_tokens
from configs.Config import Config
from peft import PromptTuningConfig


transformers.set_seed(42)


def load_config(config_path: str) -> Config:
    # Load the configuration file dynamically
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Get the Config instance from the loaded module
    config = getattr(config_module, "config", None)

    if config is None:
        raise ValueError("Config instance not found in the specified module")

    if not isinstance(config, Config):
        raise ValueError(f"The loaded instance is not of type {Config}")
    return config


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # ### Argparse
    # Here we capture the config file and any argument needed from the CLI
    # TODO: make it so you can also specify a recipe from its name using a register
    # Create the parser
    parser = argparse.ArgumentParser(description="Train the model with configurations from a specified Python configuration file.")
    # Add the config file
    parser.add_argument(
        "-c", "--config_file",
        required=True,
        help="Path to the Python file containing configuration for training. Example can be found in the configs folder."
    )

    # Add the optional dataset name
    parser.add_argument(
        "-d", "--dataset_name",
        required=False, default=None, type=str,
        help="Name or path to the dataset to run."
    )

    # Add the optional model name
    parser.add_argument(
        "-m", "--model_name",
        required=False, default=None, type=str,
        help="Name or path to the model to run."
    )

    # Add the optional number of examples
    parser.add_argument(
        "-n", "--num_examples",
        required=False, default=None, type=int,
        help="Number of examples to provide to each prompt."
    )

    # Capture the arguments
    args = parser.parse_args()
    config_file_path = args.config_file

    # Load the configuration instance
    config: Config = load_config(config_file_path)
    # Overwrite the config's variables if given
    if args.dataset_name is not None: config.dataset_name = args.dataset_name
    if args.model_name is not None: config.model_name = args.model_name
    if args.num_examples is not None: config.num_examples = args.num_examples
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Training Arguments definition
    # Here we create the training arguments that will be used for our training.
    # For more information, check out the huggingface documentation for "TrainingArguments"
    training_arguments: TrainingArguments = config.training_arguments
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # Here we define the base model we want to train.
    # Note: can be both huggingface's path or a local path
    model_name: str = config.model_name
    # An output path where to save the final model
    #   Tip: normally at the end of the training tha weights saved are the latest but they often are not
    #   the best. To solve this in the "TrainingArguments" we can specify "load_best_model_at_end=True"
    #   So that at the end of the procedure the best weights (calculated using loss on validation set) are
    #   loaded and then saved.
    output_path: str = training_arguments.output_dir
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Peft config initialization
    # For the fine-tuning we can use different peft adapters. 
    # Tip: Check out the PeftRecipe for the list of recipes available or create a custom one by extending the PeftRecipe class.
    # Here we initialize the adapter's configuration used in the following sections.
    peft_recipe: PeftRecipe = config.peft_recipe
    peft_config = PeftDispatcher(peft_recipe).get_peft_config() if peft_recipe is not None else None
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Create tokenizer and model
    # Here we create the tokenizer for the specific model
    tokenizer_recipe: TokenizerRecipe = config.tokenizer_recipe
    tokenizer = TokenizerDispatcher(tokenizer_recipe).get_tokenizer(model_name)
    # Now we create the actual model
    model_recipe: ModelRecipe = config.model_recipe
    model_template_recipe: ModelTemplateRecipe = config.model_template_recipe
    # We also initialize the quantization config (if provided)
    quantization_recipe: QuantizationRecipe = config.quantization_recipe
    quantization_config = QuantizationDispatcher(quantization_recipe).get_quantization_config() if quantization_recipe is not None else None
    # Note: during training, instead of providing the peft's config, you provide the path to the fine-tuned folder that will contain (if used) the trained adapter
    model = ModelDispatcher(model_recipe, model_template_recipe).get_model(model_name, quantization_config, peft_config)
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
    dataset_name: str = config.dataset_name
    dataset_recipe: DatasetRecipe = config.dataset_recipe
    dataset_train = DatasetDispatcher(dataset_recipe).get_dataset(dataset_name, 
                                                                  split="train", 
                                                                  postprocess_function=model_template_recipe.postprocess_function, 
                                                                  num_examples=config.num_examples)
    # TODO: The validation set will have a different set of examples, should this be fixed?
    try: 
        dataset_val = DatasetDispatcher(dataset_recipe).get_dataset(dataset_name, 
                                                 split="validation", 
                                                 postprocess_function=model_template_recipe.postprocess_function, 
                                                 num_examples=config.num_examples)
    except:
        # If you don't have a validation set available split the training set:
        dataset = DatasetDispatcher(dataset_recipe).get_dataset(dataset_name, split="train").train_test_split(test_size=config.validation_split_size)
        dataset_train, dataset_val = dataset["train"], dataset["test"]
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Response template
    # This is needed in case you want to train only on the completion of the input prompt.
    # ### Example of a prompt:
    # "What is the result of 2 + 2? Answer: 4"
    #                                      >-<
    # -> If the response template is "Answer: " then the loss will be calculated only on the following tokens in this case "4"
    #
    # Without a template, instead:
    # "What is the result of 2 + 2? Answer: 4"
    # >--------------------------------------<
    # -> The loss will be calculated on all tokens (i.e. we are training the model to also predict the question and not only the answer)
    # Here we are using an util function that given the specific dataset recipe and model recipe, finds the best response template
    # for the dataset selected and model selected.
    # [It uses the model recipe's template if it is available else the dataset recipe one and finds the best token ids to use for compatibility]
    if config.completion_only: response_template = fit_response_template_tokens(dataset_train, dataset_recipe, model_recipe, tokenizer)
    else: response_template = None
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### System template
    # This is needed if you are using `prompt tuning` as PEFT and you want to start the prepended learned tokens after a specific
    # set of words i.e. the system template of the model.
    # For example:
    # LLama2 format: <s> ... <<SYS>>  <</SYS>> ...
    #                                ^
    #                                | 
    # In this position instead of at the start of the whole prompt (before even <s>)
    # 
    # This should make the tuning better mimic the performances obtained for the typical prompt engineering done by hand
    if peft_recipe.peft_config_obj == PromptTuningConfig and config.system_tuning:
        system_template = fit_system_template_tokens(dataset_train, model_recipe, tokenizer)
        model = SystemTuning.add_proxy(model, system_template=system_template)
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### FineTuner creation and training
    # We initialize the FineTuner class and provide all the arguments needed.
    finetuner = FineTuner(
        model=model,
        tokenizer=tokenizer,
        dataset_train=dataset_train,
        dataset_validation=dataset_val,
        response_template=response_template
    )
    # Note: we also specify other kwargs as they will be given to the underlying `SFTTrainer` init.
    # For more info on the available parameters, check out the documentation for trl's SFTTrainer.
    finetuner.train(output_path, training_arguments=training_arguments, **config.finetuner_arguments)

    # We set everything to None to free up memory before the creation of new models.
    # In this example we are training only one model/dataset therefore it is not really needed.
    finetuner = None
    model = None
    tokenizer = None
    quantization_recipe = None
    # --------------------------------------------------------------------------
