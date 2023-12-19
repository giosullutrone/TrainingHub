from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Dict


class TokenizerRecipe:
    tokenizer_init_kwargs: Dict = {}
    tokenizer_kwargs: Dict = {}

    @classmethod
    def get_tokenizer(cls, tokenizer_path: str, **kwargs) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, **{**cls.tokenizer_init_kwargs, **kwargs})
        if tokenizer.pad_token is None: tokenizer.pad_token = "[PAD]"
        if tokenizer.eos_token is None: tokenizer.eos_token = "</s>"
        if tokenizer.bos_token is None: tokenizer.bos_token = "<s>"
        if tokenizer.unk_token is None: tokenizer.unk_token = "<unk>"
        for x in cls.tokenizer_kwargs: setattr(tokenizer, x, cls.tokenizer_kwargs[x])
        return tokenizer


class LLama2TokenizerRecipe(TokenizerRecipe):
    tokenizer_kwargs: Dict = {
        "padding_side": "right",
        "model_max_length": 1024
    }


class LLamaTokenizerRecipe(TokenizerRecipe):
    tokenizer_kwargs: Dict = {
        "padding_side": "left",
        "pad_token_id": 0
    }
    

class MistralTokenizerRecipe(TokenizerRecipe):
    tokenizer_kwargs: Dict = {
        # https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da#file-finetune_llama_v2-py-L210
        "padding_side": "right",
        "model_max_length": 1024
    }
