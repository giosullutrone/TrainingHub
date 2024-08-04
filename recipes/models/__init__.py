from typing import Optional
from recipes.Recipe import Recipe
from cookbooks import MODEL_COOKBOOK
from dataclasses import dataclass, field


@dataclass
class ModelRecipe(Recipe):
    model_load: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "description": "Kwargs for model load."
        }
    )
    model_config: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "description": "Kwargs for model configuration."
        }
    )

@MODEL_COOKBOOK.register()
class DefaultModelRecipe(ModelRecipe): pass
