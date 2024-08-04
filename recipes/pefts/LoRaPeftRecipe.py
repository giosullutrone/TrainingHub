from peft import LoraConfig
from typing import Dict, Union
from cookbooks import PEFT_COOKBOOK
from recipes.pefts import PeftRecipe


@PEFT_COOKBOOK.register()
class LoRaPeftRecipe(PeftRecipe):
    peft_config_obj = "LoraConfig"
    peft_config = {
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "r": 64,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }