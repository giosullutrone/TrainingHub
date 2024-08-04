from recipes.quantizations.FourBitQuantizationRecipe import FourBitQuantizationRecipe
from recipes.models.MistralModelRecipe import MistralModelRecipe
from recipes.tokenizers.MistralTokenizerRecipe import MistralTokenizerRecipe
from recipes.pefts.LoRaPeftRecipe import LoRaPeftRecipe
from recipes.datasets.PostOCRCorrection import PostOCRCorrection
from recipes.train import DefaultTrainRecipe
from configs import Config

config = Config( 
    dataset_recipe=PostOCRCorrection(
        dataset_load={"name": "english"}
    ),
    model_recipe=MistralModelRecipe(
        model_load={
            "cache_dir": "../../models",
            "use_cache": True,
            "max_position_embeddings": 2048
        },
    ),
    tokenizer_recipe=MistralTokenizerRecipe(
        tokenizer_config={
            "cache_dir": "../models",
            "model_max_length": 2048
        },
        tokenizer_system="You are a bot specialied in OCR correction, respond to the user with the corrected OCR text."        
    ),
    peft_recipe=LoRaPeftRecipe(),
    quantization_recipe=FourBitQuantizationRecipe(),
    train_recipe=DefaultTrainRecipe(
        dataset_name="PleIAs/Post-OCR-Correction",
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
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
        finetuner_arguments={
            "max_seq_length": 2048
        },
        completion_only=True,
        num_proc=12,
        shuffle=True,
        shuffle_seed=42,
    )
)
