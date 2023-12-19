from typing import Any, Dict
import torch
from transformers import BitsAndBytesConfig
  
       
class QuantizationRecipe:
    quantization_init_kwargs: Dict = {}

    @classmethod
    def get_quantization_config(cls, **kwargs) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(**{**cls.quantization_init_kwargs, **kwargs})  


class LLama2QuantizationRecipe(QuantizationRecipe):
    quantization_init_kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": False,
    }

class LLamaQuantizationRecipe(QuantizationRecipe):
    quantization_init_kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": False,
    }

class MistralQuantizationRecipe(QuantizationRecipe):
    quantization_init_kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": False,
    }
