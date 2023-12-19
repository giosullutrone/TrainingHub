import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel
from typing import Dict, Union
from peft import PeftModel, get_peft_model, PeftConfig
from recipes.ModelTemplateRecipe import LLama2ModelTemplateRecipe, NeuralModelTemplateRecipe, MistralModelTemplateRecipe, ModelTemplateRecipe


class ModelRecipe:
    model_template_recipe: ModelTemplateRecipe=None
    model_init_kwargs: Dict = {}
    model_config_kwargs: Dict = {}

    @classmethod
    def get_model(cls, model_path: str, quantization_config: BitsAndBytesConfig=None, peft_path_or_config: Union[str, PeftConfig]=None, **kwargs) -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, **{**cls.model_init_kwargs, **kwargs})
        if peft_path_or_config is not None and isinstance(peft_path_or_config, str): model = PeftModel.from_pretrained(model, peft_path_or_config)
        elif peft_path_or_config is not None and isinstance(peft_path_or_config, PeftConfig): model = get_peft_model(model, peft_path_or_config)
        for x in cls.model_config_kwargs: setattr(getattr(model, "config"), x, cls.model_config_kwargs[x])
        return model
    
    @classmethod
    def get_preprocess_function(cls) -> Dict:
        assert cls.model_template_recipe is not None, "Model template not defined for the model requested"
        return cls.model_template_recipe.get_preprocess_function()
    
    @classmethod
    def get_response_template(cls) -> str:
        assert cls.model_template_recipe is not None, "Model template not defined for the model requested"
        return cls.model_template_recipe.get_response_template()
    
    @classmethod
    def get_system_template(cls) -> str:
        assert cls.model_template_recipe is not None, "Model template not defined for the model requested"
        return cls.model_template_recipe.get_system_template()


class LLama2ModelRecipe(ModelRecipe):
    model_template_recipe = LLama2ModelTemplateRecipe
    model_init_kwargs: {
        # In case of NaN during training, change to torch.bfloat16
        "torch_dtype": torch.float16
    }
    model_config_kwargs: {
        "use_cache": False,
        "pretraining_tp": 1
    }


class LLamaModelRecipe(ModelRecipe):
    model_template_recipe = LLama2ModelTemplateRecipe
    model_init_kwargs: {
        "torch_dtype": torch.float16
    }
    model_config_kwargs: {
        "use_cache": False,
        "pretraining_tp": 1
    }


class MistralModelRecipe(ModelRecipe):
    model_template_recipe = MistralModelTemplateRecipe
    model_init_kwargs: {
        "torch_dtype": torch.float16
    }
    model_config_kwargs: {
        "use_cache": False,
        "pretraining_tp": 1
    }

class NeuralModelRecipe(MistralModelRecipe):
    model_template_recipe = NeuralModelTemplateRecipe
