from peft import LoraConfig, PeftConfig, PromptTuningConfig
from typing import Optional
from recipes.Recipe import Recipe
from cookbooks import PEFT_COOKBOOK
from dataclasses import field, dataclass


@dataclass
class PeftRecipe(Recipe):
    peft_config_obj: Optional[str] = field(
        default=None, 
        metadata={
            "description": "Kwargs for peft configuration."
        }
    )
    peft_config: Optional[dict] = field(
        default=None, 
        metadata={
            "description": "Kwargs for peft configuration."
        }
    )
    
    @property
    def peft_config_obj(self) -> PeftConfig:
        peft_classes = {
            "PeftConfig": PeftConfig,
            "LoRaConfig": LoraConfig,
            "PromptTuningConfig": PromptTuningConfig
        }
        return peft_classes[self.peft_config_obj]

@PEFT_COOKBOOK.register()
class DefaultPeftRecipe(PeftRecipe): pass
