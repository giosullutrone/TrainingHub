from typing import Dict
import torch
from cookbooks import MODEL_COOKBOOK
from recipes.models import ModelRecipe


@MODEL_COOKBOOK.register()
class MistralModelRecipe(ModelRecipe):
    model_load = {"torch_dtype": torch.bfloat16}
