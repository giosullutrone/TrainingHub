from typing import Dict, Callable, Union


class ModelTemplateRecipe:
    RESPONSE_TEMPLATE: Union[str, None] = None
    SYSTEM_TEMPLATE: Union[str, None] = None

    def __init__(self, 
                 postprocess_function: Callable[[Dict], Dict]=None, 
                 response_template: Union[str, None]=None, 
                 system_template: Union[str, None]=None) -> None:
        if postprocess_function is not None: self.postprocess_function = postprocess_function
        self._response_template = response_template if response_template is not None else self.RESPONSE_TEMPLATE
        self._system_template = system_template if system_template is not None else self.SYSTEM_TEMPLATE
    
    @staticmethod
    def postprocess_function(sample: Dict) -> Dict:
        """Note: there is no need to put <s> or </s>, it will be taken care of by the tokenizer"""
        return sample
    
    @property
    def response_template(self) -> Union[str, None]:
        return self._response_template
    
    @property
    def system_template(self) -> Union[str, None]:
        return self._system_template


class LLama2ModelTemplateRecipe(ModelTemplateRecipe):
    RESPONSE_TEMPLATE = " [/INST] "
    SYSTEM_TEMPLATE = " <<SYS>>\n "
    @staticmethod
    def postprocess_function(sample: Dict) -> Dict: return {"prompts": f'[INST] <<SYS>>\n \n<</SYS>>\n{sample["prompts"]} [/INST] '}
    
class MistralModelTemplateRecipe(ModelTemplateRecipe):
    RESPONSE_TEMPLATE = " [/INST] "
    SYSTEM_TEMPLATE = "[INST] "
    @staticmethod
    def postprocess_function(sample: Dict) -> Dict: return {"prompts": f'[INST] \n{sample["prompts"]} [/INST] '}
    
class NeuralModelTemplateRecipe(ModelTemplateRecipe):
    RESPONSE_TEMPLATE = "### Assistant:\n"
    SYSTEM_TEMPLATE = "### System:\n"
    @staticmethod
    def postprocess_function(sample: Dict) -> Dict: return {"prompts": f'### System:\n### User:\n{sample["prompts"]}\n### Assistant:\n'}