from peft import LoraConfig, PeftConfig, PromptTuningConfig, PromptTuningInit
from typing import Dict
from transformers import PreTrainedTokenizer
 

class PeftRecipe:
    peft_init_kwargs: Dict = {}
    peft_config_obj = PeftConfig

    @classmethod
    def get_peft_config(cls, **kwargs) -> PeftConfig:
        return cls.peft_config_obj(**{**cls.peft_init_kwargs, **kwargs})  


class QLoRaPeftRecipe(PeftRecipe):
    peft_config_obj = LoraConfig

    peft_init_kwargs: Dict = {
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "r": 64,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


class PromptTuningPeftRecipe(PeftRecipe):
    peft_config_obj = PromptTuningConfig

    peft_init_kwargs: Dict = {
        "prompt_tuning_init": PromptTuningInit.RANDOM,
        "prompt_tuning_init_text": "This is a really important task for me, I will tip you 200 dollars if you complete it correctly",
        "num_virtual_tokens": 32,
        "task_type": "CAUSAL_LM",
    }

    @classmethod
    def get_peft_config(cls, tokenizer_name: str, **kwargs) -> PeftConfig:
        return cls.peft_config_obj(tokenizer_name_or_path=tokenizer_name, **{**cls.peft_init_kwargs, **kwargs})
    

class PromptTuning16PeftRecipe(PeftRecipe):
    peft_config_obj = PromptTuningConfig

    peft_init_kwargs: Dict = {
        "prompt_tuning_init": PromptTuningInit.RANDOM,
        "prompt_tuning_init_text": "This is a really important task for me, I will tip you 200 dollars if you complete it correctly",
        "num_virtual_tokens": 16,
        "task_type": "CAUSAL_LM",
    }

    @classmethod
    def get_peft_config(cls, tokenizer_name: str, **kwargs) -> PeftConfig:
        return cls.peft_config_obj(tokenizer_name_or_path=tokenizer_name, **{**cls.peft_init_kwargs, **kwargs})
