from peft import PeftConfig
from typing import Dict, Union
from cookbooks import PEFT_COOKBOOK


class PeftRecipe:
    PEFT_CONFIG_OBJ: PeftConfig = PeftConfig
    PEFT_CONFIG: Union[Dict, None] = None

    def __init__(self, 
                 peft_config: Union[Dict, None]=None, 
                 peft_config_obj: PeftConfig=None) -> None:
        self._peft_config = {}
        if self.PEFT_CONFIG is not None: self._peft_config.update(self.PEFT_CONFIG)
        if peft_config is not None: self._peft_config.update(peft_config)
        self._peft_config_obj = peft_config_obj if peft_config_obj is not None else self.PEFT_CONFIG_OBJ
    
    @property
    def peft_config(self) -> Dict:
        return self._peft_config
    
    @property
    def peft_config_obj(self) -> PeftConfig:
        return self._peft_config_obj


@PEFT_COOKBOOK.register()
class DefaultPeftRecipe(PeftRecipe): pass
