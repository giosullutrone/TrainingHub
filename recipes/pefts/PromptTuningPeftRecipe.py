from peft import LoraConfig, PeftConfig, PromptTuningConfig, PromptTuningInit
from typing import Dict, Union
from cookbooks import PEFT_COOKBOOK
from recipes.pefts import PeftRecipe


@PEFT_COOKBOOK.register()
class PromptTuningPeftRecipe(PeftRecipe):
    PEFT_CONFIG_OBJ = PromptTuningConfig
    PEFT_CONFIG = {
        "prompt_tuning_init": PromptTuningInit.RANDOM,
        "num_virtual_tokens": 32,
        "task_type": "CAUSAL_LM",
    }

@PEFT_COOKBOOK.register()
class PromptTuningTextPeftRecipe(PeftRecipe):
    PEFT_CONFIG_OBJ = PromptTuningConfig
    PEFT_CONFIG = {
        "prompt_tuning_init": PromptTuningInit.TEXT,
        "prompt_tuning_init_text": "This is a really important task for me, I will tip you 200 dollars if you complete it correctly",
        "num_virtual_tokens": 32,
        "task_type": "CAUSAL_LM",
    }
