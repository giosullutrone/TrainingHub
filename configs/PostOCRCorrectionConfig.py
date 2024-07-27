from recipes.quantizations.FourBitQuantizationRecipe import FourBitQuantizationRecipe
from recipes.models.MistralModelRecipe import MistralModelRecipe, MistralModelTemplateRecipe
from recipes.tokenizers.MistralTokenizerRecipe import MistralTokenizerRecipe
from recipes.pefts.LoRaPeftRecipe import LoRaPeftRecipe
from configs.ConfigTrain import ConfigTrain
from recipes.datasets.PostOCRCorrection import PostOCRCorrection


config = ConfigTrain(
    dataset_name="PleIAs/Post-OCR-Correction",
    dataset_recipe=PostOCRCorrection(dataset_load={"name": "english"}),
    num_examples=0,
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
    model_recipe=MistralModelRecipe(
        model_load={
            "cache_dir": "../models",
            "use_cache": True,
            "max_position_embeddings": 2048
        }
    ),
    model_template_recipe=MistralModelTemplateRecipe(),
    tokenizer_recipe=MistralTokenizerRecipe(
        tokenizer_config={
            "cache_dir": "../models",
            "model_max_length": 2048
        }
    ),
    quantization_recipe=FourBitQuantizationRecipe(),
    training_arguments={
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 1,
        "optim": "adamw_torch",
        "learning_rate": 1e-5,
        "weight_decay": 0.001,
        "fp16": True,
        "bf16": False,
        "tf32": False,
        "max_grad_norm": 0.3,
        "max_steps": -1,
        "warmup_ratio": 0.03,
        "group_by_length": True,
        "lr_scheduler_type": "cosine",
        "eval_strategy": "steps",
        "logging_steps": 0.1,
        "eval_steps": 0.1,
        "save_steps": 0.1,
        "load_best_model_at_end": False,
        "save_total_limit": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 2,
        "logging_first_step": True,
        "output_dir": "./models"
    },
    peft_recipe=LoRaPeftRecipe(),
    finetuner_arguments={
        "max_seq_length": 2048
    },
    completion_only=True,
    num_proc=12
)
