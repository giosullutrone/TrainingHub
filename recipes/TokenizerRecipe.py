from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Dict, Union


class TokenizerRecipe:
    TOKENIZER_LOAD: Union[Dict, None] = None
    TOKENIZER_CONFIG: Union[Dict, None] = None

    def __init__(self, 
                 tokenizer_load: Union[Dict, None]=None, 
                 tokenizer_config: Union[Dict, None]=None) -> None:
        self._tokenizer_load = {}
        if self.TOKENIZER_LOAD is not None: self._tokenizer_load.update(self.TOKENIZER_LOAD)
        if tokenizer_load is not None: self._tokenizer_load.update(tokenizer_load)
        self._tokenizer_config = {}
        if self.TOKENIZER_CONFIG is not None: self._tokenizer_config.update(self.TOKENIZER_CONFIG)
        if tokenizer_config is not None: self._tokenizer_config.update(tokenizer_config)

    @property
    def tokenizer_load(self) -> Dict:
        return self._tokenizer_load
    
    @property
    def tokenizer_config(self) -> Dict:
        return self._tokenizer_config


class LLama2TokenizerRecipe(TokenizerRecipe):
    TOKENIZER_CONFIG = {
        "padding_side": "right",
        "model_max_length": 1024
    }

class MistralTokenizerRecipe(TokenizerRecipe):
    TOKENIZER_CONFIG = {
        # https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da#file-finetune_llama_v2-py-L210
        "padding_side": "right",
        "model_max_length": 1024
    }
