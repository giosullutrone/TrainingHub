from dataclasses import dataclass
from recipes.datasets import DatasetRecipe
from recipes.models import ModelRecipe
from recipes.tokenizers import TokenizerRecipe
from recipes.pefts import PeftRecipe
from recipes.quantizations import QuantizationRecipe
from recipes.train import TrainRecipe


@dataclass
class Config:
    dataset_recipe: DatasetRecipe
    model_recipe: ModelRecipe
    tokenizer_recipe: TokenizerRecipe
    peft_recipe: PeftRecipe
    quantization_recipe: QuantizationRecipe
    train_recipe: TrainRecipe
