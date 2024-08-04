from typing import Optional
from recipes.Recipe import Recipe
from cookbooks import PEFT_COOKBOOK
from dataclasses import field, dataclass


@dataclass
class PeftRecipe(Recipe):
    peft_config_obj: Optional[str] = field(
        default=None, 
        metadata={
            "description": "Name of the peft config to create. Check out the PeftDispatcher for more info."
        }
    )
    peft_config: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "description": "Kwargs for peft configuration."
        }
    )
    

@PEFT_COOKBOOK.register()
class DefaultPeftRecipe(PeftRecipe): pass
