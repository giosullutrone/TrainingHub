from typing import Any, Dict, Union
from cookbooks import QUANTIZATION_COOKBOOK


class QuantizationRecipe:
    QUANTIZATION_CONFIG: Union[Dict, None] = None

    def __init__(self, 
                 quantization_config: Union[Dict, None]=None) -> None:
        self._quantization_config = {}
        if self.QUANTIZATION_CONFIG is not None: self._quantization_config.update(self.QUANTIZATION_CONFIG)
        if quantization_config is not None: self._quantization_config.update(quantization_config)
    
    @property
    def quantization_config(self) -> Dict:
        return self._quantization_config


@QUANTIZATION_COOKBOOK.register()
class DefaultQuantizationRecipe(QuantizationRecipe): pass
