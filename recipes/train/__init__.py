from recipes.Recipe import Recipe
from dataclasses import field, dataclass
from cookbooks import TRAIN_COOKBOOK
from typing import Optional


@dataclass
class TrainRecipe(Recipe):
    """
    Configuration class for training by CLI.
    """

    # --------------------------------------------------------------------------
    # Here we define all the parameters for our training procedure ("train_cli.py")
    # Each parameter is a field. For each of them we specify the type, default and some useful metadata.
    # 
    # Metadata:
    #   - `description` (str): Description of the field. Used both as documentation and also by the argument parser
    #                          as information to show when calling `--help` on `train_cli.py` 
    #   - `required` (bool): Whether it is required for running `train_cli.py`. 
    #                        Note: This is done instead of removing `None` from (Union[..., None]) so that we can use also
    #                              use a starting config file for common parameters (See utils.parsers.get_config_from_argparser for the code)
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Required parameters
    # Paths
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "description": "Huggingface path or local path of the dataset.", 
            "required": True
        }
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "description": "Huggingface path or local path of the model.", 
            "required": True
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "description": "Huggingface path or local path of the tokenizer.", 
            "required": True
        }
    )
    
    # --------------------------------------------------------------------------
    # Training arguments
    finetuner_arguments: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "description": "Kwargs for finetuner configuration. For more information check out trl.SFTTrainer for available arguments."
        }
    )
    dataset_size: Optional[int] = field(
        default=None, 
        metadata={
            "description": "Number of samples to limit the training dataset to (the validation_split_size of this will be taken if specified). Default: None -> No limits"
        }
    )
    training_split: Optional[str] = field(
        default=None, 
        metadata={
            "description": "Dataset split to use for training. For more information check out: https://huggingface.co/docs/datasets/en/loading"
        }
    )
    validation_split: Optional[str] = field(
        default=None, 
        metadata={
            "description": "Dataset split to use for validation. For more information check out: https://huggingface.co/docs/datasets/en/loading"
        }
    )
    training_arguments: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "description": "Training arguments to pass to the trainer. For more information check out trl.SFTConfig for available arguments."
        }
    )
    num_examples: int = field(
        default=0, 
        metadata={
            "description": "Number of examples to provide for each prompt."
        }
    )
    completion_only: bool = field(
        default=True, 
        metadata={
            "description": "Whether to train the model only on completion (i.e. text after response template) or the full prompt."
        }
    )
    shuffle: bool = field(
        default=True, 
        metadata={
            "description": "Whether to shuffle the datasets."
        }
    )
    shuffle_seed: bool = field(
        default=42, 
        metadata={
            "description": "Seed to use for the datasets' shuffle."
        }
    )
    prompt_tuning_most_common_init: bool = field(
        default=False, 
        metadata={
            "description": "Whether to use the most common words in the dataset as starting point for prompt tuning."
        }
    )
    num_proc: int = field(
        default=None, 
        metadata={
            "description": "How many processes to use for dataset mapping."
        }
    )
    verbose: bool = field(
        default=False, 
        metadata={
            "description": "Whether to set the logger to DEBUG mode."
        }
    )

@TRAIN_COOKBOOK.register()
class DefaultTrainRecipe(TrainRecipe): pass