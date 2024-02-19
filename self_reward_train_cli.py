import transformers
from transformers import TrainingArguments
from typing import Dict, Union
from finetuners import FineTuner
from recipes import DatasetRecipe, PeftRecipe, ModelRecipe, ModelTemplateRecipe, TokenizerRecipe, QuantizationRecipe, get_examples_lm_evaluation_harness_format
from dispatchers import DatasetDispatcher, ModelDispatcher, PeftDispatcher, QuantizationDispatcher, TokenizerDispatcher
from utils import SystemTuning, fit_response_template_tokens, fit_system_template_tokens, get_config_from_argparser, most_common_words
from configs import ConfigTrain
from peft import PromptTuningConfig

import logging
logger = logging.getLogger("llm_steerability")

transformers.set_seed(42)

class NewsGenerationDatasetRecipe(DatasetRecipe):
    def preprocess_function(self, sample, examples) -> Dict:
        classes = [
            'alt.atheism',
            'comp.graphics',
            'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware',
            'comp.windows.x',
            'misc.forsale',
            'rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey',
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space',
            'soc.religion.christian',
            'talk.politics.guns',
            'talk.politics.mideast',
            'talk.politics.misc',
            'talk.religion.misc'
        ]

        def get_examples(examples) -> str:
            """
            Given a dataset to be used for examples, returns a string containing the entries and a marker for the start and end of the examples section
            """
            if examples is None: return ""
            return "Generate new news documents following the Examples.\n\n" + "\n\n".join([ex[:512] for ex in examples["text"]]) + "\n\n"
        prompt = get_examples(examples)
        prompt += f"Example:\n{sample['text']}\nTopic:{sample['label_text']}"
        return {"prompts": prompt, "labels": ""}


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
    dataset_name: str = "SetFit/20_newsgroups"
    dataset_recipe: DatasetRecipe = NewsGenerationDatasetRecipe()
    model_template_recipe: ModelTemplateRecipe = config.model_template_recipe
    dataset_support, dataset_train = DatasetDispatcher(dataset_recipe).get_support_and_tuning_dataset(dataset_name, 
                                                                                                      split="train", 
                                                                                                      num_examples=1,
                                                                                                      postprocess_function=model_template_recipe.postprocess_function,
                                                                                                      eos_token="",
                                                                                                      include_labels_inside_text=True,
                                                                                                      num_proc=config.num_proc,
                                                                                                      dynamic_examples=True)
    try:
        _, dataset_val = DatasetDispatcher(dataset_recipe).get_support_and_tuning_dataset(dataset_name, 
                                                                                          split="validation", 
                                                                                          dataset_support=dataset_support,
                                                                                          num_examples=1,
                                                                                          postprocess_function=model_template_recipe.postprocess_function, 
                                                                                          eos_token="",
                                                                                          include_labels_inside_text=True,
                                                                                          num_proc=config.num_proc,
                                                                                          dynamic_examples=True)
    except:
        # If you don't have a validation set available, split the training set
        assert config.validation_split_size is not None, "No validation set is available but validation split size has not been specified"
        logger.info("Validation dataset not found, splitting train into two different sets.")
        if config.dynamic_examples: logger.warning("Using dynamic examples with splitted sets currently leads to small amount of data leakage between training and validation set,\
                                                   use with caution as the validation set will likely overperform.")
        dataset = dataset_train.train_test_split(test_size=config.validation_split_size)
        dataset_train, dataset_val = dataset["train"], dataset["test"]

    if config.training_size: dataset_train = dataset_train.select(range(config.training_size))

    # Logger
    logger.debug(f'First prompt for training set:\n{dataset_train["text"][0]}')
    logger.debug(f'First prompt for validation set:\n{dataset_val["text"][0]}')
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
    if peft_recipe is not None and peft_recipe.peft_config_obj == PromptTuningConfig and config.system_tuning:
        system_template = fit_system_template_tokens(dataset_train, model_template_recipe, tokenizer)
        model = SystemTuning.add_proxy(model, system_template=system_template, tokenizer=tokenizer)
        logger.debug(f'Added system tuning modification as requested')
    # --------------------------------------------------------------------------
    
    def sample(sample):
        old_padding_size = tokenizer.padding_side
        tokenizer.padding_side = "left"
        result = {"text": [tokenizer.decode(x, skip_special_tokens=True)[0] for x in model.generate(
            input_ids=tokenizer.encode(sample["text"], padding=False, return_tensors="pt").to(model.device),
            generation_config=transformers.GenerationConfig(
                num_return_sequences=4,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=300
            )
        )]}
        tokenizer.padding_side = old_padding_size
        return result

    dataset_generated = dataset_train.select(range(1)).map(sample, desc="Generating new samples")

    print("######################\n\n", dataset_generated["text"], "######################\n\n", dataset_generated["text"][0])
    for t in dataset_generated["text"]:
        print("Generated: ", t.split(model_template_recipe.model_response_template)[-1], "\\n")

    """
    outputs = model.generate(
        input_ids=tokenizer.encode("[INST] Generate new prompts following the examples.\nExample 1 Instruction: You are given a science question (easy-level) and four answer options (associated with “A”, “B”, “C”, “D”). Your task is to find the correct answer based on scientific facts, knowledge, and reasoning. Do not generate anything else apart from one of the following characters: ‘A’, ‘B, ‘C’, ‘D’. There is only one correct answer for each question. Input: Which part of a bicycle BEST moves in a circle? (A) Seat (B) Frame (C) Foot pedal (D) Kickstand Constraints: The output should be one of the following characters: ‘A’, ‘B, ‘C’, ‘D’. Example 2 Instruction: You are given a negative review and your task is to convert it to a positive review by one or more making minimal changes. Avoid changing the context of the review. Input: we stood there in shock, because we never expected this. Constraints: None. Example 3 [/INST]", return_tensors="pt").to(model.device),
        generation_config=transformers.GenerationConfig(
            num_return_sequences=4,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=150
        )
    )
    for output in outputs:
        generated_texts = tokenizer.decode(output, skip_special_tokens=True)
        print("\n###########\n", generated_texts)
    """