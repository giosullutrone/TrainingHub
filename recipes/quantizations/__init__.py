from typing import Optional
from recipes.Recipe import Recipe
from cookbooks import QUANTIZATION_COOKBOOK
from dataclasses import field, dataclass

@dataclass
class QuantizationRecipe(Recipe):
    quantization_config: Optional[dict] = field(
        default=None, 
        metadata={
            "description": "Kwargs for quantization configuration."
        }
    )

@QUANTIZATION_COOKBOOK.register()
class DefaultQuantizationRecipe(QuantizationRecipe): pass
