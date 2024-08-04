from recipes.pefts import PeftRecipe
from peft import PeftConfig, LoraConfig, PromptTuningConfig


class PeftDispatcher:
    def __init__(self, peft_recipe: PeftRecipe) -> None:
        self._peft_recipe = peft_recipe

    def get_peft_config(self) -> PeftConfig:
        peft_classes = {
            "PeftConfig": PeftConfig,
            "LoraConfig": LoraConfig,
            "PromptTuningConfig": PromptTuningConfig
        }
        return peft_classes[self.peft_recipe.peft_config_obj](**self.peft_recipe.peft_config) 
    
    @property
    def peft_recipe(self) -> PeftRecipe:
        return self._peft_recipe