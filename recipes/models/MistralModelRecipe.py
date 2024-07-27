from typing import Dict
import torch
from cookbooks import MODEL_COOKBOOK
from cookbooks import MODEL_TEMPLATE_COOKBOOK
from recipes.models import ModelRecipe, ModelTemplateRecipe


@MODEL_COOKBOOK.register()
class MistralModelRecipe(ModelRecipe):
    MODEL_LOAD = {"torch_dtype": torch.bfloat16}

@MODEL_TEMPLATE_COOKBOOK.register()
class MistralModelTemplateRecipe(ModelTemplateRecipe):
    RESPONSE_TEMPLATE = " [/INST]"
    SYSTEM_TEMPLATE = "[INST] "
    @staticmethod
    def postprocess_function(sample: Dict) -> Dict: return {"prompts": f'[INST] \n{sample["prompts"]} [/INST]'}