from recipes.PeftRecipe import PeftRecipe
from peft import PeftConfig


class PeftDispatcher:
    def __init__(self, peft_recipe: PeftRecipe) -> None:
        self._peft_recipe = peft_recipe

    def get_peft_config(self) -> PeftConfig:
        return self.peft_recipe.peft_config_obj(**self.peft_recipe.peft_config) 
    
    @property
    def peft_recipe(self) -> PeftRecipe:
        return self._peft_recipe