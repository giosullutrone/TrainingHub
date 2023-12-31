from typing import Dict, Callable


class ModelTemplateRecipe:
    """
    Note: there is no need to put <s> or </s>, it will be taken care of by the tokenizer
    """
    @classmethod
    def get_postprocess_function(cls) -> Callable[..., Dict]:
        raise NotImplementedError()
    
    @classmethod
    def get_response_template(cls) -> str:
        raise NotImplementedError()
    
    @classmethod
    def get_system_template(cls) -> str:
        raise NotImplementedError()
    

class LLama2ModelTemplateRecipe(ModelTemplateRecipe):
    @classmethod
    def get_postprocess_function(cls) -> Callable[..., Dict]:
        return lambda sample: {"prompts": f'[INST] <<SYS>>\n \n<</SYS>>\n{sample["prompts"]} [/INST] '}
    
    @classmethod
    def get_response_template(cls) -> str:
        return " [/INST] "

    @classmethod
    def get_system_template(cls) -> str:
        return " <<SYS>>\n "


class MistralModelTemplateRecipe(ModelTemplateRecipe):
    @classmethod
    def get_postprocess_function(cls) -> Callable[..., Dict]:
        return lambda sample: {"prompts": f'[INST] \n{sample["prompts"]} [/INST] '}
    
    @classmethod
    def get_response_template(cls) -> str:
        return " [/INST] "

    @classmethod
    def get_system_template(cls) -> str:
        return "[INST] "
    

class NeuralModelTemplateRecipe(ModelTemplateRecipe):
    @classmethod
    def get_postprocess_function(cls) -> Callable[..., Dict]:
        return lambda sample: {"prompts": f'### System:\n### User:\n{sample["prompts"]}\n### Assistant:\n'}
    
    @classmethod
    def get_response_template(cls) -> str:
        return "### Assistant:\n"

    @classmethod
    def get_system_template(cls) -> str:
        return "### System:\n"
    
