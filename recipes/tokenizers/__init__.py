from typing import Optional
from recipes.Recipe import Recipe
from cookbooks import TOKENIZER_COOKBOOK
from dataclasses import field, dataclass


@dataclass
class TokenizerRecipe(Recipe):
    tokenizer_load: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "description": "Kwargs for tokenizer load."
        }
    )
    tokenizer_config: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "description": "Kwargs for tokenizer configuration."
        }
    )
    tokenizer_chat_template: Optional[str] = field(
        default=None,
        metadata={
            "description": "Chat template to use for prompt generation."
        }
    )

@TOKENIZER_COOKBOOK.register()
class DefaultTokenizerRecipe(TokenizerRecipe): pass
