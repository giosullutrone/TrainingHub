from enum import Enum
from recipes.ModelRecipe import LLama2ModelRecipe, MistralModelRecipe, LLamaModelRecipe, ModelRecipe, NeuralModelRecipe
from recipes.QuantizationRecipe import LLama2QuantizationRecipe, LLamaQuantizationRecipe, MistralQuantizationRecipe, QuantizationRecipe
from recipes.TokenizerRecipe import LLama2TokenizerRecipe, LLamaTokenizerRecipe, MistralTokenizerRecipe, TokenizerRecipe
from typing import Tuple


class ModelEnum(Enum):
    LLAMA2 = "llama2"
    MISTRAL = "mistral"
    NEURAL = "neural"


class DispatcherRecipe:
    @staticmethod
    def get_recipes(model_type: ModelEnum) -> Tuple[TokenizerRecipe, ModelRecipe, QuantizationRecipe]:
        if model_type == ModelEnum.LLAMA2: return (LLama2TokenizerRecipe, LLama2ModelRecipe, LLama2QuantizationRecipe)
        if model_type == ModelEnum.MISTRAL: return (MistralTokenizerRecipe, MistralModelRecipe, MistralQuantizationRecipe)
        if model_type == ModelEnum.NEURAL: return (MistralTokenizerRecipe, NeuralModelRecipe, MistralQuantizationRecipe)
        raise Exception("Model type not found")