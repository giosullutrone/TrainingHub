from typing import Dict, Union, Callable
from cookbooks import MODEL_COOKBOOK
from cookbooks import MODEL_TEMPLATE_COOKBOOK


class ModelRecipe:
    MODEL_LOAD: Union[Dict, None] = None
    MODEL_CONFIG: Union[Dict, None] = None

    def __init__(self, model_load: Union[Dict, None]=None, model_config: Union[Dict, None]=None) -> None:
        self._model_load = {}
        if self.MODEL_LOAD is not None: self._model_load.update(self.MODEL_LOAD)
        if model_load is not None: self._model_load.update(model_load)
        self._model_config = {}
        if self.MODEL_CONFIG is not None: self._model_config.update(self.MODEL_CONFIG)
        if model_config is not None: self._model_config.update(model_config)

    @property
    def model_load(self) -> Dict:
        return self._model_load

    @property
    def model_config(self) -> Dict:
        return self._model_config

class ModelTemplateRecipe:
    RESPONSE_TEMPLATE: Union[str, None] = None
    SYSTEM_TEMPLATE: Union[str, None] = None

    def __init__(self, 
                 postprocess_function: Callable[[Dict], Dict]=None, 
                 model_response_template: Union[str, None]=None, 
                 system_template: Union[str, None]=None) -> None:
        if postprocess_function is not None: self.postprocess_function = postprocess_function
        self._model_response_template = model_response_template if model_response_template is not None else self.RESPONSE_TEMPLATE
        self._system_template = system_template if system_template is not None else self.SYSTEM_TEMPLATE
    
    @staticmethod
    def postprocess_function(sample: Dict) -> Dict:
        """Note: there is no need to put <s> or </s>, it will be taken care of by the tokenizer"""
        return sample
    
    @property
    def model_response_template(self) -> Union[str, None]:
        return self._model_response_template
    
    @property
    def system_template(self) -> Union[str, None]:
        return self._system_template

@MODEL_COOKBOOK.register()
class DefaultModelRecipe(ModelRecipe): pass

@MODEL_TEMPLATE_COOKBOOK.register()
class DefaultModelTemplateRecipe(ModelTemplateRecipe): pass