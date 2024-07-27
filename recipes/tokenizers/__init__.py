from typing import Dict, Union
from cookbooks import TOKENIZER_COOKBOOK


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


@TOKENIZER_COOKBOOK.register()
class DefaultTokenizerRecipe(TokenizerRecipe): pass
