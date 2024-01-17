from peft import LoraConfig, PeftConfig, PromptTuningConfig, PromptTuningInit
from typing import Dict, Union
from cookbooks import PEFT_COOKBOOK


class PeftRecipe:
    PEFT_CONFIG_OBJ: PeftConfig = PeftConfig
    PEFT_CONFIG: Union[Dict, None] = None

    def __init__(self, 
                 peft_config: Union[Dict, None]=None, 
                 peft_config_obj: PeftConfig=None) -> None:
        self._peft_config = {}
        if self.PEFT_CONFIG is not None: self._peft_config.update(self.PEFT_CONFIG)
        if peft_config is not None: self._peft_config.update(peft_config)
        self._peft_config_obj = peft_config_obj if peft_config_obj is not None else self.PEFT_CONFIG_OBJ
    
    @property
    def peft_config(self) -> Dict:
        return self._peft_config
    
    @property
    def peft_config_obj(self) -> PeftConfig:
        return self._peft_config_obj


@PEFT_COOKBOOK.register()
class DefaultPeftRecipe(PeftRecipe): pass

@PEFT_COOKBOOK.register()
class QLoRaPeftRecipe(PeftRecipe):
    PEFT_CONFIG_OBJ = LoraConfig
    PEFT_CONFIG = {
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "r": 64,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

@PEFT_COOKBOOK.register()
class PromptTuningPeftRecipe(PeftRecipe):
    PEFT_CONFIG_OBJ = PromptTuningConfig
    PEFT_CONFIG = {
        "prompt_tuning_init": PromptTuningInit.RANDOM,
        "prompt_tuning_init_text": "This is a really important task for me, I will tip you 200 dollars if you complete it correctly",
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

    def __init__(self, 
                 tokenizer_name_or_path: str,
                 peft_config: Union[Dict, None] = None, 
                 peft_config_obj: PeftConfig = None) -> None:
        super().__init__({**peft_config, "tokenizer_name_or_path": tokenizer_name_or_path}, peft_config_obj)
    
@PEFT_COOKBOOK.register()
class PromptTuning16PeftRecipe(PeftRecipe):
    PEFT_CONFIG_OBJ = PromptTuningConfig
    PEFT_CONFIG = {
        "prompt_tuning_init": PromptTuningInit.RANDOM,
        "num_virtual_tokens": 16,
        "task_type": "CAUSAL_LM",
    }
