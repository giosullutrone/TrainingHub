from recipes.tokenizers import TokenizerRecipe
from transformers import PreTrainedTokenizer, AutoTokenizer


class TokenizerDispatcher:
    def __init__(self, tokenizer_recipe: TokenizerRecipe) -> None:
        self._tokenizer_recipe = tokenizer_recipe

    def get_tokenizer(self, tokenizer_path: str) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, **self.tokenizer_recipe.tokenizer_load)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        for x in self.tokenizer_recipe.tokenizer_config: setattr(tokenizer, x, self.tokenizer_recipe.tokenizer_config[x])
        return tokenizer
    
    @property
    def tokenizer_recipe(self) -> TokenizerRecipe:
        return self._tokenizer_recipe
