from typing import Dict, Union
from cookbooks import TOKENIZER_COOKBOOK
from recipes.tokenizers import TokenizerRecipe

@TOKENIZER_COOKBOOK.register()
class MistralTokenizerRecipe(TokenizerRecipe):
    TOKENIZER_CONFIG = {
        # https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da#file-finetune_llama_v2-py-L210
        "padding_side": "right",
        "model_max_length": 1024
    }