import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel
from typing import Dict, Union
from peft import PeftModel, get_peft_model, PeftConfig
from recipes.ModelTemplateRecipe import ModelTemplateRecipe
from recipes.ModelRecipe import ModelRecipe


class ModelDispatcher:
    def __init__(self, model_recipe: ModelRecipe) -> None:
        self._model_recipe = model_recipe

    def get_model(self, model_path: str, quantization_config: Union[BitsAndBytesConfig, None]=None, peft_path_or_config: Union[str, PeftConfig]=None) -> PreTrainedModel:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, **self.model_recipe.model_load)
        if peft_path_or_config is not None and isinstance(peft_path_or_config, str): model = PeftModel.from_pretrained(model, peft_path_or_config)
        elif peft_path_or_config is not None and isinstance(peft_path_or_config, PeftConfig): model = get_peft_model(model, peft_path_or_config)
        for x in self.model_recipe.model_config: setattr(getattr(model, "config"), x, self.model_recipe.model_config[x])
        return model
    
    @property
    def model_recipe(self) -> ModelRecipe:
        return self._model_recipe
