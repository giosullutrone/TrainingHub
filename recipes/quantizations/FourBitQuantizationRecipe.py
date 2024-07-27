from typing import Any, Dict, Union
import torch
from cookbooks import QUANTIZATION_COOKBOOK
from recipes.quantizations import QuantizationRecipe


@QUANTIZATION_COOKBOOK.register()
class FourBitQuantizationRecipe(QuantizationRecipe):
    QUANTIZATION_CONFIG = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": False,
    }