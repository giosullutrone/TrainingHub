from typing import Dict, Union
import torch


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


class LLama2ModelRecipe(ModelRecipe):
    # MODEL_LOAD = {"torch_dtype": torch.bfloat16}
    pass

class MistralModelRecipe(ModelRecipe):
    # MODEL_LOAD = {"torch_dtype": torch.float16}
    pass