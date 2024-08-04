from dataclasses import dataclass, field
from recipes.datasets import DatasetRecipe
from recipes.models import ModelRecipe
from recipes.tokenizers import TokenizerRecipe
from recipes.pefts import PeftRecipe
from recipes.quantizations import QuantizationRecipe
from recipes.train import TrainRecipe
from cookbooks import *

@dataclass
class Config:
    # --------------------------------------------------------------------------
    # Here we define all the parameters for our training procedure ("train_cli.py")
    # Each parameter is a field. For each of them we specify the type, default and some useful metadata.
    # 
    # Metadata:
    #   - `cookbook` (CookBook): Cookbook to use to get the `recipe` if it was specified by string. 
    #                            For example --model_recipe "MistralModelRecipe", the string "MistralModelRecipe" would be used to get the 
    #                            class from the given cookbook.
    # --------------------------------------------------------------------------
    
    dataset_recipe: DatasetRecipe = field(metadata={"cookbook": DATASET_COOKBOOK})
    model_recipe: ModelRecipe = field(metadata={"cookbook": MODEL_COOKBOOK})
    tokenizer_recipe: TokenizerRecipe = field(metadata={"cookbook": TOKENIZER_COOKBOOK})
    peft_recipe: PeftRecipe = field(metadata={"cookbook": PEFT_COOKBOOK})
    quantization_recipe: QuantizationRecipe = field(metadata={"cookbook": QUANTIZATION_COOKBOOK})
    train_recipe: TrainRecipe = field(metadata={"cookbook": TRAIN_COOKBOOK})
