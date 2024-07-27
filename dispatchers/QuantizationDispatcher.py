from transformers import BitsAndBytesConfig
from recipes.quantizations import QuantizationRecipe


class QuantizationDispatcher:
    def __init__(self, quantization_recipe: QuantizationRecipe) -> None:
        self._quantization_recipe = quantization_recipe

    def get_quantization_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(**self.quantization_recipe.quantization_config)
    
    @property
    def quantization_recipe(self) -> QuantizationRecipe:
        return self._quantization_recipe