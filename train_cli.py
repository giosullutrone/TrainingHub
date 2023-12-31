import argparse
import importlib.util
import transformers
from transformers import TrainingArguments
from typing import Dict, Union
from finetuner import FineTuner
from recipes.DatasetRecipe import DatasetRecipe
from recipes.PeftRecipe import PeftRecipe
from recipes.ModelRecipe import ModelRecipe
from recipes.TokenizerRecipe import TokenizerRecipe
from recipes.QuantizationRecipe import QuantizationRecipe
from utils.ProxySystemPromptTuning import ProxySystemPromptTuning
from utils import fit_response_template_tokens, fit_system_template_tokens
from torch.utils.data import random_split


transformers.set_seed(42)


def load_config(config_file_path):
    spec = importlib.util.spec_from_file_location("config_module", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with configurations from a specified Python configuration file.")
    parser.add_argument(
        "-c", "--config_file",
        required=True,
        help="Path to the Python file containing configuration for training. Example can be found in the configs folder."
    )

    args = parser.parse_args()
    config_file_path = args.config_file

    # Load the configuration module
    config_module = load_config(config_file_path)

    # --------------------------------------------------------------------------
    # ### Training Arguments definition
    # Here we create the training arguments that will be used for our training
    # Note: these are just examples and they can be changed for the specific use case.
    # For more information, check out the huggingface documentation for "TrainingArguments"
    training_arguments: TrainingArguments = config_module.training_arguments
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # Here we define the base model we want to train.
    # Note: can be both huggingface's path or a local path
    model_name: str = config_module.model_name
    # The class of the model (Needed because we will use a dispatcher in this example, check next code blocks)
    # model_class = ModelEnum.MISTRAL
    # An output path where to save the final model
    # Tip: normally at the end of the training tha weights saved are the latest but they often are not
    # the best. To solve this in the "TrainingArguments" we specified "load_best_model_at_end=True"
    # So that at the end of the procedure the best weights (calculated using loss on validation set) are
    # loaded and then saved.
    output_path: str = training_arguments["output_dir"]
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Peft config initialization
    # For the fine-tuning of this example we are going to train the LoRa adapter
    # and not the whole model. Here we initialize the adapter's configuration used
    # in the following section.
    peft_recipe: PeftRecipe = config_module.peft_recipe
    peft_config = peft_recipe.get_peft_config() if peft_recipe is not None else None
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Create tokenizer and model
    # Here we use the dispatcher to get all the recipes needed by specifying the model class using "ModelEnum"
    # Tip: you can also directly import the recipes that you want from "cookbook.recipes"
    # tokenizer_recipe, model_recipe, quantization_recipe = DispatcherRecipe.get_recipes(model_class)
    tokenizer_recipe: TokenizerRecipe = config_module.tokenizer_recipe
    model_recipe: ModelRecipe = config_module.model_recipe
    quantization_recipe: QuantizationRecipe = config_module.quantization_recipe
    # We also initialize the quantization config to use to reduce memory usage
    quantization_config = quantization_recipe.get_quantization_config() if quantization_recipe is not None else None
    # We also give the "peft_path_or_config" to apply the "peft" adapter to the model
    # Note: during training, instead of providing the config class, we provide the path to the adapter's folder
    model = model_recipe.get_model(model_name, quantization_config=quantization_config, peft_path_or_config=peft_config, **config_module.model_arguments)
    tokenizer = tokenizer_recipe.get_tokenizer(model_name, **config_module.tokenizer_arguments)
    # --------------------------------------------------------------------------
    
    
    # --------------------------------------------------------------------------
    # ### Dataset
    # We load the datasets from huggingface using the specific recipe
    # For more datasets check out recipes.DatasetRecipe, if you don't find the one that you want
    # follow the dataset recipe creation guide # TODO: Write a guide for the creation of new recipes
    #
    # Here we are creating a train and validation datasets with zero-shot (n_example=0) and
    # adapted to the template for the specific model (postprocess_function=model_recipe.get_postprocess_function())
    dataset_name: str = config_module.dataset_name
    dataset_recipe: DatasetRecipe = config_module.dataset_recipe
    dataset_train = dataset_recipe.get_dataset(dataset_name, split="train", postprocess_function=model_recipe.get_postprocess_function(), **config_module.dataset_arguments)
    try:
        dataset_val = dataset_recipe.get_dataset(dataset_name, split="validation", postprocess_function=model_recipe.get_postprocess_function(), **config_module.dataset_arguments)
    except:
        # If you don't have a validation set available split the training set:
        dataset = dataset_recipe.get_dataset(dataset_name, split="train").train_test_split(test_size=config_module.validation_split_size)
        dataset_train, dataset_val = dataset["train"], dataset["test"]


    # --------------------------------------------------------------------------
    # ### Response template
    # This is needed in case you want to train only on the completion of the input prompt
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
    response_template = fit_response_template_tokens(dataset_train, dataset_recipe, model_recipe, tokenizer)
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### System template
    # This is needed if you are using Prompt tuning as PEFT and you want to start the prepended learned tokens after a specific
    # set of words i.e. the system template of the model.
    # For example:
    # LLama2 format: <s> ... <<SYS>>  <</SYS>> ...
    #                                ^
    #                                | 
    # In this position instead of at the start of the whole prompt (before even <s>)
    # 
    # This should make the tuning better mimic the performances obtained for the typical prompt engineering done by hand
    # system_template = fit_system_template_tokens(dataset_train, model_recipe, tokenizer)
    # model = ProxySystemPromptTuning.add_proxy(model, system_template=system_template)
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
    # Note: we can also specify other kwargs as they will be given to the underlying "SFTTrainer.__init__"
    # For more info on the available parameters, check out the documentation for "trl.SFTTrainer"
    finetuner.train(output_path, training_arguments=training_arguments, **config_module.finetuner_arguments)

    # We set everything to None to free up memory before the creation of new models
    finetuner = None
    model = None
    tokenizer = None
    quantization_recipe = None
    # --------------------------------------------------------------------------
