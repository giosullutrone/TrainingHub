from peft import LoraConfig
from typing import Dict, Union
from cookbooks import PEFT_COOKBOOK
from recipes.pefts import PeftRecipe


@PEFT_COOKBOOK.register()
class LoRaPeftRecipe(PeftRecipe):
    peft_config_obj = "LoraConfig"
    peft_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        "modules_to_save": None,
    }